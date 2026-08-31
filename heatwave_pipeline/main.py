from .config import *
from .utils.helpers import configure_console, log, save_csv
from .data.hmis import (
    load_hmis,
    extract_hmis_indicators,
    build_hmis_baseline
)
from .data.weather import (
    fetch_weather,
    fetch_forecast
)
from .data.hmis_monthly import load_monthly_hmis, build_monthly_vulnerability, attach_monthly_vulnerability
from .dataset.builder import (
    prepare_ml_dataset,
    print_quality_report
)
from .models.trainer import train_models
from .features.health_risk import add_health_risk_features
from .risk.scorer import score_health_impact
from .forecast.risk_forecast import build_5day_forecast
from .gis.map_builder import export_gis_payload
import pandas as pd
import json


def main():

    # ========================================================
    # START
    # ========================================================

    configure_console()

    log("\n" + "=" * 70)
    log("ASSAM HEATWAVE HEALTH-IMPACT PIPELINE")
    log("=" * 70)

    log(f"HMIS resource ID: {HMIS_RESOURCE_ID}")
    log(f"HMIS file: {HMIS_FILE}")
    log(
        f"Historical period: "
        f"{START_DATE} -> {END_DATE}"
    )
    log(
        f"Districts: {len(DISTRICTS)}"
    )

    # ========================================================
    # 1. LOAD HMIS
    # ========================================================

    log("\n" + "=" * 70)
    log("1. LOADING HMIS HEALTH DATA")
    log("=" * 70)

    hmis_raw = load_hmis()

    if hmis_raw is None:

        log(
            "[WARNING] HMIS data could not be loaded."
        )

        hmis_indicators = None
        hmis_baseline = None

    else:

        hmis_indicators = (
            extract_hmis_indicators(
                hmis_raw
            )
        )

        if hmis_indicators is not None:

            hmis_baseline = (
                build_hmis_baseline(
                    hmis_indicators
                )
            )

        else:

            hmis_baseline = None

    # ========================================================
    # 2. HISTORICAL WEATHER
    # ========================================================

    log("\n" + "=" * 70)
    log("2. LOADING HISTORICAL WEATHER")
    log("=" * 70)

    weather = fetch_weather()

    if weather is None or weather.empty:

        log(
            "[FATAL] Historical weather "
            "could not be obtained."
        )

        return

    log(
        f"[OK] Weather rows: "
        f"{len(weather):,}"
    )

    # ========================================================
    # 3. FORECAST
    # ========================================================

    log("\n" + "=" * 70)
    log("3. FETCHING 5-DAY FORECAST")
    log("=" * 70)

    try:

        # Always request the newest forecast when the pipeline is run.
        forecast = fetch_forecast(force_refresh=True)

        if forecast is not None:

            log(
                f"[OK] Forecast rows: "
                f"{len(forecast):,}"
            )

        else:

            log(
                "[WARNING] Forecast unavailable."
            )

    except Exception as e:

        log(
            f"[WARNING] Forecast failed: {e}"
        )

    # ========================================================
    # 4. CREATE ML DATASET
    # ========================================================

    log("\n" + "=" * 70)
    log("4. PREPARING ML DATASET")
    log("=" * 70)

    dataset = prepare_ml_dataset(
        weather,
        hmis_baseline
    )

    # Prefer real monthly HMIS observations when the sub-district/monthly files
    # are present. Never fabricate monthly values from the static baseline.
    monthly_raw = load_monthly_hmis()
    if monthly_raw is not None:
        monthly_vulnerability = build_monthly_vulnerability(monthly_raw)
        if monthly_vulnerability is not None and not monthly_vulnerability.empty:
            dataset = attach_monthly_vulnerability(dataset, monthly_vulnerability)
            save_csv(monthly_vulnerability, "hmis_monthly_vulnerability.csv")
            log(f"[OK] Monthly HMIS vulnerability rows: {len(monthly_vulnerability):,}")
        else:
            log("[WARNING] Monthly HMIS files found but could not be parsed; retaining static HMIS baseline.")
    else:
        log("[INFO] No monthly/sub-district HMIS CSVs found in hmis_monthly/; static baseline retained. No synthetic monthly values are created.")

    if dataset is None or dataset.empty:

        log(
            "[FATAL] ML dataset could not "
            "be created."
        )

        return

    # ========================================================
    # 4B. ADD HEALTH VULNERABILITY FEATURES
    # ========================================================

    dataset = add_health_risk_features(dataset, hmis_baseline)

    # ========================================================
    # 5. SAVE ML DATASET
    # ========================================================

    log("\n" + "=" * 70)
    log("5. SAVING ML DATASET")
    log("=" * 70)

    save_csv(
        dataset,
        "heatwave_ml_dataset.csv"
    )

    # ========================================================
    # 6. QUALITY REPORT
    # ========================================================

    print_quality_report(
        dataset
    )

    # ========================================================
    # 7. TRAIN HEATWAVE FORECAST MODELS
    # ========================================================

    log("\n" + "=" * 70)
    log("7. TRAINING HEATWAVE FORECAST MODELS")
    log("=" * 70)

    model_result = None
    try:

        model_result = train_models(
            dataset,
            target="heatwave_next_3d"
        )

        log(
            f"[OK] Best model: {model_result['best_model_name']}"
        )

    except Exception as e:

        log(
            f"[WARNING] ML training skipped: {e}"
        )

    # ========================================================
    # 8. BUILD HEAT-HEALTH IMPACT RISK OUTPUT
    # ========================================================

    log("\n" + "=" * 70)
    log("8. BUILDING HEAT-HEALTH IMPACT RISK")
    log("=" * 70)

    risk_output = score_health_impact(dataset)
    save_csv(risk_output, "heat_health_risk_historical.csv")
    log("[OK] Created heat-health impact risk scores.")
    log("[INFO] These are operational impact estimates, not synthetic mortality labels.")

    # ========================================================
    # 9. BUILD 5-DAY FORECAST, GIS AND ALERT OUTPUTS
    # ========================================================
    if 'forecast' in locals() and forecast is not None and not forecast.empty:
        try:
            from .dataset.builder import standardize_weather_columns
            from .features.thermal import add_thermal_features

            fc = standardize_weather_columns(forecast.copy())
            if "date" not in fc.columns:
                raise ValueError("Forecast data does not contain a date column.")

            fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
            fc = fc.dropna(subset=["date"]).copy()
            if fc.empty:
                raise ValueError("Forecast data contains no valid dates.")

            fc = add_thermal_features(fc)
            fc = add_health_risk_features(fc, hmis_baseline)
            dashboard = build_5day_forecast(
                fc,
                model=(model_result or {}).get("best_model"),
                feature_columns=None
            )

            save_csv(dashboard, "district_5day_risk_forecast.csv")
            top = dashboard.sort_values("health_impact_risk_score", ascending=False).head(10)
            save_csv(top, "top_risk_districts.csv")

            export_gis_payload(dashboard, BASE_DIR / "gis_risk_payload.json")

            summary = {
                "generated_rows": int(len(dashboard)),
                "districts": int(dashboard["district"].nunique()) if "district" in dashboard.columns else 0,
                "alert_counts": dashboard["health_impact_risk_level"].value_counts().to_dict() if "health_impact_risk_level" in dashboard.columns else {},
                "max_risk": float(dashboard["health_impact_risk_score"].max()) if "health_impact_risk_score" in dashboard.columns else None
            }
            (BASE_DIR / "dashboard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            log("[OK] Created district_5day_risk_forecast.csv")
            log("[OK] Created top_risk_districts.csv")
            log("[OK] Created gis_risk_payload.json")
            log("[OK] Created dashboard_summary.json")

        except Exception as e:
            log(f"[WARNING] Forecast risk output skipped: {e}")

    # ========================================================
    # 7. OUTPUT FILE CHECK
    # ========================================================

    log("\n" + "=" * 70)
    log("OUTPUT FILES")
    log("=" * 70)

    output_files = [
        "hmis_health_raw.csv",
        "hmis_health_indicators.csv",
        "hmis_district_baseline.csv",
        "historical_weather_partial.csv",
        "historical_weather_assam.csv",
        "forecast_5_day_assam.csv",
        "heatwave_ml_dataset.csv",
        "heat_health_risk_historical.csv",
        "district_5day_risk_forecast.csv",
        "top_risk_districts.csv",
        "dashboard_summary.json",
        "gis_risk_payload.json"
    ]

    for filename in output_files:

        path = BASE_DIR / filename

        if path.exists():

            log(
                f"  [OK] {filename}"
            )

        else:

            log(
                f"  [--] {filename}"
            )

    # ========================================================
    # FINISH
    # ========================================================

    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE")
    log("=" * 70)

    log(
        "No synthetic mortality labels were created."
    )

    log(
        "Weather-based thermal targets are available "
        "for ML development."
    )


if __name__ == "__main__":
    main()