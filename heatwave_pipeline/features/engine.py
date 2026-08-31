"""Feature engineering for heatwave forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.helpers import log


ID_COLUMNS = ["date", "district"]
TARGET_COLUMNS = ["heatwave_next_3d", "heatwave_next_5d"]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns or "district" not in df.columns:
        raise ValueError("Dataset must contain date and district columns.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "district"])
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    g = df.groupby("district", group_keys=False)

    lag_sources = {
        "temperature_max_c": "temperature_max",
        "temperature_min_c": "temperature_min",
        "relative_humidity_mean_pct": "humidity",
        "wind_speed_kmh": "wind_speed",
        "rainfall_mm": "rainfall",
        "heat_index_c": "heat_index",
        "wet_bulb_c": "wet_bulb",
        "wbgt_estimated_c": "wbgt",
        "human_thermal_stress_index": "htsi",
    }

    for source, prefix in lag_sources.items():
        if source not in df.columns:
            continue
        for lag in (1, 2, 3, 7):
            df[f"{prefix}_lag_{lag}d"] = g[source].shift(lag)

    # Recent exposure windows.
    for window in (3, 5, 7):
        if "human_thermal_stress_index" in df.columns:
            shifted = g["human_thermal_stress_index"].shift(1)
            df[f"htsi_prev_{window}d_mean"] = (
                shifted.groupby(df["district"])
                .rolling(window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )
        if "wbgt_estimated_c" in df.columns:
            shifted = g["wbgt_estimated_c"].shift(1)
            df[f"wbgt_prev_{window}d_mean"] = (
                shifted.groupby(df["district"])
                .rolling(window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # Cyclical calendar signals.
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)

    return df


def prepare_model_features(df: pd.DataFrame, target: str = "heatwave_next_3d"):
    """Return leakage-controlled, forecast-compatible weather features.

    The live forecast has no HMIS outcome observations and no future lags, so the
    heatwave model is deliberately restricted to weather/climatology variables that
    can be computed from the forecast itself.
    """
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    work = add_lag_features(df)
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target]).copy()

    preferred = [
        "temperature_max_c", "temperature_min_c", "temperature_mean_c",
        "relative_humidity_mean_pct", "wind_speed_kmh", "rainfall_mm",
        "heat_index_c", "wet_bulb_c", "wbgt_estimated_c",
        "temperature_anomaly_c", "wbgt_anomaly_c",
        "district_tmax_p90_c", "district_tmax_p95_c",
        "district_wbgt_p90_c", "district_wbgt_p95_c",
        "district_hi_p90_c", "district_hi_p95_c",
        "district_tmin_p90_c", "district_tmin_p95_c",
        "tmax_above_p90", "tmax_above_p95",
        "wbgt_above_p90", "wbgt_above_p95",
        "heat_index_above_p90", "heat_index_above_p95",
        "hot_day_streak", "extreme_wbgt_streak", "hot_night_3d_count",
        "temperature_3d_mean_c", "temperature_5d_mean_c",
        "wbgt_3d_mean_c", "wbgt_5d_mean_c",
        "heat_index_3d_mean_c", "heat_index_5d_mean_c",
        "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
    ]
    feature_columns = [c for c in preferred if c in work.columns]
    if len(feature_columns) < 10:
        raise RuntimeError("Too few forecast-compatible weather features are available.")

    X = work[feature_columns].copy()
    y = work[target].astype(int)
    return work, X, y, feature_columns

def chronological_split(work: pd.DataFrame, train_fraction=0.70, valid_fraction=0.15):
    """Split by time; no random shuffling."""
    work = work.sort_values("date").reset_index(drop=True)
    n = len(work)
    train_end = int(n * train_fraction)
    valid_end = int(n * (train_fraction + valid_fraction))

    train = work.iloc[:train_end].copy()
    valid = work.iloc[train_end:valid_end].copy()
    test = work.iloc[valid_end:].copy()

    return train, valid, test


def log_feature_summary(work: pd.DataFrame, feature_columns: list[str], target: str):
    log("")
    log("=" * 70)
    log("MODEL FEATURE SUMMARY")
    log("=" * 70)
    log(f"Rows: {len(work):,}")
    log(f"Features: {len(feature_columns):,}")
    log(f"Target: {target}")
    log(f"Positive cases: {(work[target] == 1).sum():,}")
    log(f"Negative cases: {(work[target] == 0).sum():,}")
