from __future__ import annotations

import numpy as np
import pandas as pd


def wet_bulb_stull(temp_c, rh):
    """Approximate wet-bulb temperature (Stull 2011)."""
    T = pd.to_numeric(temp_c, errors="coerce")
    R = pd.to_numeric(rh, errors="coerce").clip(1, 100)
    return (
        T * np.arctan(0.151977 * np.sqrt(R + 8.313659))
        + np.arctan(T + R)
        - np.arctan(R - 1.676331)
        + 0.00391838 * (R ** 1.5) * np.arctan(0.023101 * R)
        - 4.686035
    )


def heat_index_celsius(temp_c, rh):
    """NOAA-style heat-index approximation for warm/humid conditions."""
    T = pd.to_numeric(temp_c, errors="coerce")
    R = pd.to_numeric(rh, errors="coerce").clip(0, 100)
    Tf = T * 9 / 5 + 32

    hi_f = (
        -42.379 + 2.04901523 * Tf + 10.14333127 * R
        - 0.22475541 * Tf * R - 0.00683783 * Tf**2
        - 0.05481717 * R**2 + 0.00122874 * Tf**2 * R
        + 0.00085282 * Tf * R**2 - 0.00000199 * Tf**2 * R**2
    )
    hi_c = (hi_f - 32) * 5 / 9
    return hi_c.where((T >= 27) & (R >= 40), T)


def add_thermal_features(df):
    """Build heat-health features from canonical daily weather columns."""
    df = df.copy()

    # Forecast CSVs may load dates as strings. Convert before any .dt access.
    if "date" not in df.columns:
        raise ValueError("Weather data must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        df = df.dropna(subset=["date"]).copy()
    if df.empty:
        raise ValueError("No valid datetime values were found in the 'date' column.")

    required = [
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "relative_humidity_mean_pct",
        "wind_speed_kmh",
        "solar_radiation_mj_m2",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    for col in required + ["rainfall_mm", "pressure_hpa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["heat_index_c"] = heat_index_celsius(
        df["temperature_max_c"],
        df["relative_humidity_mean_pct"],
    )

    df["wet_bulb_c"] = wet_bulb_stull(
        df["temperature_mean_c"],
        df["relative_humidity_mean_pct"],
    )

    # This is a transparent outdoor WBGT proxy because the daily sources
    # do not supply measured globe temperature. Do not label it as measured WBGT.
    solar = pd.to_numeric(
        df["solar_radiation_mj_m2"], errors="coerce"
    ).fillna(0)
    wind = pd.to_numeric(
        df["wind_speed_kmh"], errors="coerce"
    ).fillna(1)
    T = df["temperature_mean_c"]
    R = df["relative_humidity_mean_pct"].fillna(50)

    globe_proxy = (
        T
        + 0.015 * np.sqrt(solar.clip(lower=0) * 1000 / 86.4)
        - 0.08 * np.sqrt(wind.clip(lower=0.5))
        - 0.005 * (R - 50)
    )

    df["wbgt_estimated_c"] = (
        0.7 * df["wet_bulb_c"]
        + 0.2 * globe_proxy
        + 0.1 * T
    )

    df["temp_humidity_interaction"] = (
        df["temperature_max_c"] * df["relative_humidity_mean_pct"] / 100
    )
    df["humidity_deficit_pct"] = 100 - df["relative_humidity_mean_pct"]

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    g = df.groupby("district", group_keys=False)

    for w in (3, 5, 7):
        df[f"temperature_{w}d_mean_c"] = g["temperature_max_c"].transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        )
        df[f"wbgt_{w}d_mean_c"] = g["wbgt_estimated_c"].transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        )
        df[f"heat_index_{w}d_mean_c"] = g["heat_index_c"].transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        )
        if "rainfall_mm" in df.columns:
            df[f"rainfall_{w}d_sum_mm"] = g["rainfall_mm"].transform(
                lambda s: s.rolling(w, min_periods=w).sum()
            )

    df["temperature_monthly_normal_c"] = df.groupby(
        ["district", "month"]
    )["temperature_max_c"].transform("median")
    df["temperature_anomaly_c"] = (
        df["temperature_max_c"] - df["temperature_monthly_normal_c"]
    )

    df["wbgt_monthly_normal_c"] = df.groupby(
        ["district", "month"]
    )["wbgt_estimated_c"].transform("median")
    df["wbgt_anomaly_c"] = (
        df["wbgt_estimated_c"] - df["wbgt_monthly_normal_c"]
    )

    # Seasonal local climatology: heat stress is relative to what is unusual for
    # this district and month, not relative to the entire year.
    for value_col, p90_col, p95_col in [
        ("temperature_max_c", "district_tmax_p90_c", "district_tmax_p95_c"),
        ("wbgt_estimated_c", "district_wbgt_p90_c", "district_wbgt_p95_c"),
        ("heat_index_c", "district_hi_p90_c", "district_hi_p95_c"),
        ("temperature_min_c", "district_tmin_p90_c", "district_tmin_p95_c"),
    ]:
        grouped = df.groupby(["district", "month"])[value_col]
        df[p90_col] = grouped.transform(lambda s: s.quantile(0.90))
        df[p95_col] = grouped.transform(lambda s: s.quantile(0.95))

    df["tmax_above_p90"] = (df["temperature_max_c"] > df["district_tmax_p90_c"]).astype("int8")
    df["tmax_above_p95"] = (df["temperature_max_c"] > df["district_tmax_p95_c"]).astype("int8")
    df["wbgt_above_p90"] = (df["wbgt_estimated_c"] > df["district_wbgt_p90_c"]).astype("int8")
    df["wbgt_above_p95"] = (df["wbgt_estimated_c"] > df["district_wbgt_p95_c"]).astype("int8")
    df["heat_index_above_p90"] = (df["heat_index_c"] > df["district_hi_p90_c"]).astype("int8")
    df["heat_index_above_p95"] = (df["heat_index_c"] > df["district_hi_p95_c"]).astype("int8")

    def streak(s):
        groups = s.eq(0).cumsum()
        return (s.groupby(groups).cumcount() + 1).where(s.eq(1), 0)

    # A heatwave day is a local extreme in either dry-bulb or outdoor thermal stress.
    # Using both protects against a hot/dry day and a humid/low-wind day being missed.
    df["heatwave_day_flag"] = (
        (df["tmax_above_p95"] == 1) | (df["wbgt_above_p95"] == 1)
    ).astype("int8")
    df["hot_day_streak"] = g["heatwave_day_flag"].transform(streak)
    df["extreme_wbgt_streak"] = g["wbgt_above_p95"].transform(streak)

    df["hot_night"] = (df["temperature_min_c"] > df["district_tmin_p90_c"]).astype("int8")
    df["hot_night_3d_count"] = g["hot_night"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )

    # HTSI (0-100): seasonal percentile burden + anomaly + persistence.
    # Percentile bands are local and month-specific, which avoids treating Assam's
    # cool-season and pre-monsoon temperatures as if they had the same baseline.
    def band_burden(value, p90, p95):
        width = (p95 - p90).replace(0, np.nan)
        return ((value - p90) / width).clip(0, 1).fillna(0)

    wbgt_burden = band_burden(df["wbgt_estimated_c"], df["district_wbgt_p90_c"], df["district_wbgt_p95_c"])
    hi_burden = band_burden(df["heat_index_c"], df["district_hi_p90_c"], df["district_hi_p95_c"])
    tmax_burden = band_burden(df["temperature_max_c"], df["district_tmax_p90_c"], df["district_tmax_p95_c"])
    hot_night_burden = band_burden(df["temperature_min_c"], df["district_tmin_p90_c"], df["district_tmin_p95_c"])
    anomaly_burden = (
        (df["wbgt_anomaly_c"] - 1.0) / 4.0
    ).clip(0, 1).fillna(0)
    persistence_burden = (df["hot_day_streak"] / 3.0).clip(0, 1).fillna(0)

    df["human_thermal_stress_index"] = (
        100 * (
            0.35 * wbgt_burden
            + 0.18 * hi_burden
            + 0.17 * tmax_burden
            + 0.10 * hot_night_burden
            + 0.10 * anomaly_burden
            + 0.10 * persistence_burden
        )
    ).round(1).clip(0, 100)

    df["thermal_risk_level"] = pd.cut(
        df["human_thermal_stress_index"],
        bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=["Low", "Moderate", "High", "Very High", "Extreme"],
    )

    return df
