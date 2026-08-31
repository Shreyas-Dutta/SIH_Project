from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.helpers import log
from ..features.thermal import add_thermal_features
from ..features.engine import add_lag_features


def normalize_district_name(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    value = " ".join(value.replace("-", " ").split())
    aliases = {
        "kamrup m": "kamrup metropolitan",
        "kamrup metropolitan": "kamrup metropolitan",
        "kamrup metro": "kamrup metropolitan",
        "kamrup r": "kamrup rural",
        "kamrup rural": "kamrup rural",
        "marigaon": "morigaon",
        "morigaon": "morigaon",
        "sibsagar": "sivasagar",
        "sivasagar": "sivasagar",
    }
    return aliases.get(value, value)


def standardize_weather_columns(weather):
    """Map source weather columns to the canonical names used downstream."""
    df = weather.copy()
    mapping = {
        "temperature_max_c": ["temperature_2m_max", "temperature_max_c"],
        "temperature_min_c": ["temperature_2m_min", "temperature_min_c"],
        "temperature_mean_c": ["temperature_2m_mean", "temperature_mean_c"],
        "relative_humidity_mean_pct": ["relative_humidity_2m_mean", "relative_humidity_mean_pct"],
        "relative_humidity_max_pct": ["relative_humidity_2m_max", "relative_humidity_max_pct"],
        "relative_humidity_min_pct": ["relative_humidity_2m_min", "relative_humidity_min_pct"],
        "wind_speed_kmh": ["wind_speed_10m_mean", "wind_speed_10m_max", "wind_speed_kmh"],
        "wind_direction_deg": ["wind_direction_10m_dominant", "wind_direction_deg"],
        "solar_radiation_mj_m2": ["shortwave_radiation_sum", "shortwave_radiation_mean", "solar_radiation_mj_m2"],
        "rainfall_mm": ["precipitation_sum", "rain_sum", "rainfall_mm"],
        "apparent_temperature_max_c": ["apparent_temperature_max", "apparent_temperature_max_c"],
        "apparent_temperature_min_c": ["apparent_temperature_min", "apparent_temperature_min_c"],
        "apparent_temperature_mean_c": ["apparent_temperature_mean", "apparent_temperature_mean_c"],
        "et0_mm": ["et0_fao_evapotranspiration", "et0_mm"],
        "wind_gust_kmh": ["wind_gusts_10m_max", "wind_gust_kmh"],
        "pressure_hpa": ["surface_pressure_mean", "pressure_hpa"],
    }
    for target, candidates in mapping.items():
        if target in df.columns and df[target].notna().any():
            continue
        for source in candidates:
            if source in df.columns and df[source].notna().any():
                df[target] = pd.to_numeric(df[source], errors="coerce")
                break
    return df


def merge_weather_hmis(weather, hmis_baseline):
    if weather is None or weather.empty:
        return None

    result = standardize_weather_columns(weather)
    if "district" not in result.columns:
        raise ValueError("Weather data must contain district.")
    result["district"] = result["district"].map(normalize_district_name)

    if hmis_baseline is None or hmis_baseline.empty:
        return result

    health = hmis_baseline.copy()
    if "district" not in health.columns:
        log("[WARNING] HMIS baseline has no district column; skipping HMIS merge.")
        return result
    health["district"] = health["district"].map(normalize_district_name)
    health = health.drop_duplicates("district", keep="first")

    weather_districts = set(result["district"].dropna().unique())
    health_districts = set(health["district"].dropna().unique())
    matched = weather_districts & health_districts
    log(f"[MERGE CHECK] Weather districts: {len(weather_districts)}")
    log(f"[MERGE CHECK] HMIS districts: {len(health_districts)}")
    log(f"[MERGE CHECK] Matched districts: {len(matched)}")

    result = result.merge(
        health,
        on="district",
        how="left",
        validate="many_to_one",
    )
    return result

def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def remove_empty_columns(df):
    """Remove columns that contain no usable information at all."""
    df = df.copy()
    empty = [c for c in df.columns if df[c].isna().all()]

    if empty:
        log("[CLEAN] Removing completely empty columns:")
        for c in empty:
            log(f"  - {c}")
        df = df.drop(columns=empty)

    return df


def remove_constant_columns(df, protected=None):
    """Remove non-informative constant columns, except identifiers/targets."""
    df = df.copy()
    protected = set(protected or [])

    drop = []
    for c in df.columns:
        if c in protected:
            continue
        if df[c].nunique(dropna=True) <= 1:
            drop.append(c)

    if drop:
        log("[CLEAN] Removing constant columns:")
        for c in drop:
            log(f"  - {c}")
        df = df.drop(columns=drop)

    return df


def convert_numeric_columns(df):
    df = df.copy()

    protected = {
        "district",
        "date",
        "thermal_risk_level",
        "weather_source",
        "forecast_source",
    }

    for col in df.columns:
        if col in protected:
            continue

        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() >= 0.75:
                df[col] = converted

    return df


def identify_health_columns(df):
    keywords = [
        "hmis",
        "health",
        "hospital",
        "admission",
        "outpatient",
        "inpatient",
        "patient",
        "bed",
        "doctor",
        "nurse",
        "facility",
        "birth",
        "death",
        "disease",
        "malaria",
        "dengue",
        "diarrhea",
        "respiratory",
        "cardiac",
        "stroke",
    ]
    return [
        c for c in df.columns
        if any(k in str(c).lower() for k in keywords)
        and _is_numeric(df[c])
    ]


def impute_numeric_features(df):
    """Impute remaining numeric feature gaps without leaving whole empty columns."""
    df = df.copy()

    id_cols = {"district", "date"}
    target_prefixes = ("future_", "heatwave_next_")
    numeric = [
        c for c in df.columns
        if c not in id_cols
        and _is_numeric(df[c])
        and not str(c).startswith(target_prefixes)
    ]

    # Add missingness flags only for columns that actually have gaps.
    missing_cols = [c for c in numeric if int(df[c].isna().sum()) > 0]
    if missing_cols:
        missing_flags = pd.DataFrame(
            {f"{c}_was_missing": df[c].isna().astype("int8") for c in missing_cols},
            index=df.index,
        )
        df = pd.concat([df, missing_flags], axis=1)

    # Weather/physical variables: interpolate by district where ordered by date.
    if "district" in df.columns and "date" in df.columns:
        df = df.sort_values(["district", "date"]).reset_index(drop=True)
        for c in numeric:
            if c not in df.columns or df[c].notna().all():
                continue
            df[c] = df.groupby("district")[c].transform(
                lambda s: s.interpolate(
                    method="linear",
                    limit_direction="both",
                )
            )

    # Remaining gaps: median of the observed data.
    for c in numeric:
        if c not in df.columns or df[c].isna().sum() == 0:
            continue

        median = df[c].median()
        if pd.notna(median):
            df[c] = df[c].fillna(median)

    return df


def add_vulnerability_features(df):
    """
    Do NOT fabricate demographic variables.
    Only retain vulnerability fields if a real data source populated them.
    """
    df = df.copy()

    # Delete old placeholder columns if they are completely empty.
    placeholders = [
        "elderly_density",
        "outdoor_worker_density",
        "population_density",
        "hospital_density",
        "healthcare_capacity",
    ]

    for c in placeholders:
        if c in df.columns and df[c].isna().all():
            df = df.drop(columns=[c])

    return df


def add_future_weather_targets(df):
    if "district" not in df.columns or "date" not in df.columns:
        return df

    df = df.copy().sort_values(["district", "date"]).reset_index(drop=True)
    g = df.groupby("district", group_keys=False)

    if "temperature_max_c" in df.columns:
        for h in range(1, 6):
            df[f"future_tmax_t_plus_{h}"] = g["temperature_max_c"].shift(-h)

    if "wbgt_estimated_c" in df.columns:
        for h in range(1, 6):
            df[f"future_wbgt_t_plus_{h}"] = g["wbgt_estimated_c"].shift(-h)

    if "human_thermal_stress_index" in df.columns:
        for h in range(1, 6):
            df[f"future_htsi_t_plus_{h}"] = g["human_thermal_stress_index"].shift(-h)

    wbgt_cols = [f"future_wbgt_t_plus_{h}" for h in range(1, 6) if f"future_wbgt_t_plus_{h}" in df.columns]
    if wbgt_cols:
        df["future_3day_max_wbgt"] = df[wbgt_cols[:3]].max(axis=1)
        df["future_5day_max_wbgt"] = df[wbgt_cols].max(axis=1)

        # Weather-derived alert targets, not mortality targets.
        # A heatwave day is an extreme in either Tmax or WBGT relative to the
        # district-month climatology. Require persistence (2 of next 3, 3 of next 5).
        if "district_wbgt_p95_c" in df.columns and "district_tmax_p95_c" in df.columns:
            future_flags3 = []
            for h in range(1, 4):
                tmax_flag = df[f"future_tmax_t_plus_{h}"] > df["district_tmax_p95_c"]
                wbgt_flag = df[f"future_wbgt_t_plus_{h}"] > df["district_wbgt_p95_c"]
                # A heatwave day is extreme in dry-bulb temperature OR thermal stress.
                future_flags3.append((tmax_flag | wbgt_flag).astype("float"))
            f3 = pd.concat(future_flags3, axis=1)
            df["heatwave_next_3d"] = (f3.sum(axis=1) >= 2).astype("float")
            df.loc[f3.isna().any(axis=1), "heatwave_next_3d"] = np.nan

            future_flags5 = []
            for h in range(1, 6):
                tmax_flag = df[f"future_tmax_t_plus_{h}"] > df["district_tmax_p95_c"]
                wbgt_flag = df[f"future_wbgt_t_plus_{h}"] > df["district_wbgt_p95_c"]
                future_flags5.append((tmax_flag | wbgt_flag).astype("float"))
            f5 = pd.concat(future_flags5, axis=1)
            df["heatwave_next_5d"] = (f5.sum(axis=1) >= 3).astype("float")
            df.loc[f5.isna().any(axis=1), "heatwave_next_5d"] = np.nan

    # Remove last five rows per district where future targets do not exist.
    if "future_5day_max_wbgt" in df.columns:
        df = df[df["future_5day_max_wbgt"].notna()].copy()

    return df


def prepare_ml_dataset(weather, hmis_baseline):
    log("\n" + "=" * 70)
    log("PREPARING ML DATASET")
    log("=" * 70)

    df = merge_weather_hmis(
        weather,
        hmis_baseline,
    )

    if df is None or df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "district"])
    df = df.drop_duplicates(["district", "date"])

    df = convert_numeric_columns(df)

    # Build thermal features from verified weather variables.
    df = add_thermal_features(df)
    df = add_vulnerability_features(df)
    df = add_future_weather_targets(df)

    # Add leakage-safe temporal model features.
    df = add_lag_features(df)

    # Remove all-empty columns before imputation.
    df = remove_empty_columns(df)

    # Never allow old placeholder columns back into the feature matrix.
    df = impute_numeric_features(df)

    # Drop columns that are still completely empty after processing.
    df = remove_empty_columns(df)

    # Protect identifiers and genuine target columns from constant-column cleanup.
    protected = {
        "district",
        "date",
        "heatwave_next_3d",
        "heatwave_next_5d",
        "future_3day_max_wbgt",
        "future_5day_max_wbgt",
    }

    df = remove_constant_columns(
        df,
        protected=protected,
    )

    return df.reset_index(drop=True)


def print_quality_report(df):
    log("\n" + "=" * 70)
    log("FINAL ML DATASET QUALITY REPORT")
    log("=" * 70)

    log(f"Rows:       {len(df):,}")
    log(f"Columns:    {len(df.columns):,}")

    if "district" in df.columns:
        log(f"Districts:  {df['district'].nunique():,}")

    if "date" in df.columns:
        log(
            f"Date range: {df['date'].min().date()} -> "
            f"{df['date'].max().date()}"
        )

    empty = [c for c in df.columns if df[c].isna().all()]
    log(f"Empty columns: {len(empty)}")

    if empty:
        for c in empty:
            log(f"  - {c}")

    missing_cells = int(df.isna().sum().sum())
    log(f"Missing cells: {missing_cells:,}")

    if "heatwave_next_3d" in df.columns:
        log("\n3-day heatwave target:")
        log(df["heatwave_next_3d"].value_counts(dropna=False).to_string())

    if "heatwave_next_5d" in df.columns:
        log("\n5-day heatwave target:")
        log(df["heatwave_next_5d"].value_counts(dropna=False).to_string())
