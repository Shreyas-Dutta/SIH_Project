from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from ..features.thermal import add_thermal_features
from ..features.engine import add_lag_features
from ..risk.scorer import score_health_impact

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "heatwave_forecast_compact.joblib"
META_PATH = BASE_DIR / "models" / "heatwave_forecast_compact_meta.json"
CLIMO_PATH = BASE_DIR / "heatwave_climatology.csv"


def _load_climatology():
    if not CLIMO_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLIMO_PATH)


def _apply_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Use historical month/district thresholds instead of forecast-only percentiles."""
    cl = _load_climatology()
    if cl.empty or "month" not in df.columns:
        return df
    out = df.copy()
    out["month"] = pd.to_datetime(out["date"]).dt.month
    exact = cl[cl["district"] != "__ALL__"].copy()
    global_cl = cl[cl["district"] == "__ALL__"].copy()
    out = out.merge(exact, on=["district", "month"], how="left", suffixes=("", "_cl"))
    out = out.merge(global_cl, on=["month"], how="left", suffixes=("", "_global"))
    for base in ["tmax_normal_c", "tmax_p90_c", "tmax_p95_c", "wbgt_normal_c", "wbgt_p90_c", "wbgt_p95_c", "tmin_p90_c", "tmin_p95_c", "hi_p90_c", "hi_p95_c"]:
        out[base] = pd.to_numeric(out.get(base), errors="coerce")
        out[base] = out[base].fillna(pd.to_numeric(out.get(f"{base}_global"), errors="coerce"))
    out["temperature_monthly_normal_c"] = out["tmax_normal_c"]
    out["temperature_anomaly_c"] = out["temperature_max_c"] - out["tmax_normal_c"]
    out["wbgt_monthly_normal_c"] = out["wbgt_normal_c"]
    out["wbgt_anomaly_c"] = out["wbgt_estimated_c"] - out["wbgt_normal_c"]
    out["district_tmax_p90_c"] = out["tmax_p90_c"]
    out["district_tmax_p95_c"] = out["tmax_p95_c"]
    out["district_wbgt_p90_c"] = out["wbgt_p90_c"]
    out["district_wbgt_p95_c"] = out["wbgt_p95_c"]
    out["district_tmin_p90_c"] = out["tmin_p90_c"]
    out["district_tmin_p95_c"] = out["tmin_p95_c"]
    out["district_hi_p90_c"] = out["hi_p90_c"]
    out["district_hi_p95_c"] = out["hi_p95_c"]
    out["tmax_above_p90"] = (out["temperature_max_c"] > out["district_tmax_p90_c"]).astype("int8")
    out["tmax_above_p95"] = (out["temperature_max_c"] > out["district_tmax_p95_c"]).astype("int8")
    out["wbgt_above_p90"] = (out["wbgt_estimated_c"] > out["district_wbgt_p90_c"]).astype("int8")
    out["wbgt_above_p95"] = (out["wbgt_estimated_c"] > out["district_wbgt_p95_c"]).astype("int8")
    out["heat_index_above_p90"] = (out["heat_index_c"] > out["district_hi_p90_c"]).astype("int8")
    out["heat_index_above_p95"] = (out["heat_index_c"] > out["district_hi_p95_c"]).astype("int8")
    out["hot_night"] = (out["temperature_min_c"] > out["district_tmin_p90_c"]).astype("int8")
    out = out.sort_values(["district", "date"]).reset_index(drop=True)
    out["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(out["date"]).dt.month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(out["date"]).dt.month / 12.0)
    out["dayofyear_sin"] = np.sin(2 * np.pi * pd.to_datetime(out["date"]).dt.dayofyear / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * pd.to_datetime(out["date"]).dt.dayofyear / 365.25)
    g = out.groupby("district", group_keys=False)
    def streak(s):
        groups = s.eq(0).cumsum()
        return (s.groupby(groups).cumcount() + 1).where(s.eq(1), 0)
    out["hot_day_streak"] = g["tmax_above_p90"].transform(streak)
    out["extreme_wbgt_streak"] = g["wbgt_above_p95"].transform(streak)
    out["hot_night_3d_count"] = g["hot_night"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    return out


def add_forecast_heatwave_probability(df: pd.DataFrame, model=None, feature_columns=None, threshold=None) -> pd.DataFrame:
    """Predict next-3-day heatwave likelihood from forecast-available weather features."""
    out = df.copy()
    if model is None:
        if not MODEL_PATH.exists():
            out["heatwave_probability_pct"] = np.nan
            out["heatwave_model_status"] = "model unavailable"
            return out
        model = joblib.load(MODEL_PATH)
    out = _apply_climatology(out)
    # Build exactly the same lag/persistence features used during training.
    # These are historical/past-only and therefore available for a live forecast.
    out = add_lag_features(out)
    out = out.sort_values(["district", "date"]).reset_index(drop=True)
    g = out.groupby("district", group_keys=False)
    for w in (3, 5):
        out[f"temperature_{w}d_mean_c"] = g["temperature_max_c"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        out[f"wbgt_{w}d_mean_c"] = g["wbgt_estimated_c"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        out[f"heat_index_{w}d_mean_c"] = g["heat_index_c"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    features = feature_columns or metadata.get("features", list(getattr(model, "feature_names_in_", [])))
    # Use the threshold selected automatically on validation data during training.
    # A caller may explicitly override it for experiments.
    if threshold is None:
        threshold = float(metadata.get("threshold", 0.5))
    for c in features:
        if c not in out.columns:
            out[c] = np.nan
    p = model.predict_proba(out[features])[:, 1]
    ml_prob = np.clip(p, 0, 1)
    # Add a transparent forecast-event evidence term. It is not called a second
    # probability; it stabilizes the score when 2+ of the next 3 forecast days
    # cross the local extreme threshold.
    gflags = out.groupby("district")["wbgt_above_p95"]
    future_flag_frames = [gflags.shift(-h) for h in (1, 2, 3)]
    future_flag_df = pd.concat(future_flag_frames, axis=1)
    future_count = future_flag_df.notna().sum(axis=1).replace(0, np.nan)
    event_evidence = (future_flag_df.sum(axis=1) / future_count).fillna(out["wbgt_above_p95"]).to_numpy()
    # Keep the operational probability calibrated to the trained classifier.
    # Event evidence is reported separately and must not silently alter the
    # thresholded ML prediction. This prevents an extra source of false positives.
    blended = ml_prob
    out["heatwave_probability_pct"] = (100 * np.clip(ml_prob, 0, 1)).round(1)
    out["heatwave_probability_raw_pct"] = (100 * ml_prob).round(1)
    out["heatwave_model_threshold"] = float(threshold)
    out["heatwave_model_prediction"] = (blended >= float(threshold))
    out["heatwave_model_status"] = "live weather model + train-only climatology + auto validation threshold"
    # Forward-looking event flag: at least 2 of the next 3 forecast days exceed
    # the local WBGT 95th percentile. This matches the training target direction.
    future_flag_df = pd.concat([gflags.shift(-h) for h in (1, 2, 3)], axis=1)
    out["heatwave_forecast_event"] = (future_flag_df.sum(axis=1) >= 2).fillna(False)
    return out


def build_5day_forecast(forecast_df, output_path=None, model=None, feature_columns=None, threshold=None):
    df = add_thermal_features(forecast_df)
    df = add_forecast_heatwave_probability(df, model=model, feature_columns=feature_columns, threshold=threshold)
    df = score_health_impact(df, heat_probability_col="heatwave_probability_pct")
    keep = [c for c in [
        "date", "district", "heatwave_probability_pct", "heatwave_forecast_event",
        "human_thermal_stress_index", "thermal_risk_level", "hot_day_streak",
        "wbgt_above_p95", "health_vulnerability_score", "health_impact_risk_score",
        "health_impact_risk_level", "advisory_actions", "fetched_at_ist"
    ] if c in df]
    out = df[keep].sort_values(["date", "health_impact_risk_score"], ascending=[True, False])
    if output_path:
        out.to_csv(output_path, index=False)
    return out
