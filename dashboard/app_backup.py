"""
Assam Heat-Health Early Warning Dashboard — v2 (redesigned)

Layout (matches the wireframe):
  [ Location ▾ ]   [ Date ▾ ]        MAP        [ Category ▾ ]
  --------------------------------------------------------------
  Heatwave Chance        |
  Human Thermal Stress   |     Color-coded GIS risk map
  Thermal Risk Level     |     (RED / ORANGE / YELLOW / GREEN)
  Health Impact Score    |
  Health Impact Level    |  [ Location ]  [ Time ]  [ Alert ]
  --------------------------------------------------------------
  Filtered risk table
  Alert feed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Paths — dashboard/app.py lives one level below the project root, so the
# `heatwave_pipeline` package (a sibling of this file's parent) needs to be
# put on sys.path before it can be imported.
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heatwave_pipeline.config import DISTRICTS  # district -> (lat, lon)
from heatwave_pipeline.data.weather import fetch_forecast
from heatwave_pipeline.dataset.builder import standardize_weather_columns
from heatwave_pipeline.features.health_risk import add_health_risk_features
from heatwave_pipeline.features.thermal import add_thermal_features
from heatwave_pipeline.risk.scorer import score_health_impact

st.set_page_config(
    page_title="Assam Heat-Health Early Warning",
    layout="wide",
    page_icon="🌡️",
)

# --------------------------------------------------------------------------
# Color scales
# --------------------------------------------------------------------------
RISK_COLORS = {
    "GREEN": "#2ecc71",
    "YELLOW": "#f1c40f",
    "ORANGE": "#e67e22",
    "RED": "#e74c3c",
    "UNKNOWN": "#95a5a6",
}
RISK_ORDER = ["RED", "ORANGE", "YELLOW", "GREEN", "UNKNOWN"]

THERMAL_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Very High": "#e74c3c",
    "Extreme": "#8e44ad",
    "nan": "#95a5a6",
}

# --------------------------------------------------------------------------
# Data loading with a graceful fallback: use the pre-built pipeline output
# if it exists, otherwise compute the same risk features live from the raw
# 5-day forecast so the dashboard never depends on a manual pipeline run.
# --------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Fetching the latest weather forecast...")
def load_risk_data(refresh_token: int = 0):
    """Fetch the newest 5-day forecast and calculate risk from it.

    The dashboard deliberately does not use district_5day_risk_forecast.csv
    as its primary source because that file can become stale between pipeline
    runs. The raw forecast is refreshed from Open-Meteo every cache interval
    (5 minutes) and immediately on the manual refresh button.
    """
    del refresh_token  # only used to invalidate Streamlit's cache

    forecast = fetch_forecast(force_refresh=True)
    if forecast is None or forecast.empty:
        return pd.DataFrame(), "Open-Meteo (latest fetch failed)", False

    forecast = standardize_weather_columns(forecast)
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
    forecast = forecast.dropna(subset=["date", "district"]).copy()

    baseline_path = ROOT / "hmis_district_baseline.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None

    computed = add_thermal_features(forecast)
    computed = add_health_risk_features(computed, baseline)
    computed = score_health_impact(computed)

    keep = [c for c in [
        "date", "district", "human_thermal_stress_index", "thermal_risk_level",
        "hot_day_streak", "wbgt_above_p95", "health_vulnerability_score",
        "health_impact_risk_score", "health_impact_risk_level", "advisory_actions",
        "fetched_at_ist",
    ] if c in computed.columns]
    computed = computed[keep].sort_values(["date", "health_impact_risk_score"], ascending=[True, False])

    fetched_at = (
        str(computed["fetched_at_ist"].dropna().iloc[0])
        if "fetched_at_ist" in computed.columns and computed["fetched_at_ist"].notna().any()
        else pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    )
    return _finalize(computed), f"Open-Meteo live forecast (fetched {fetched_at})", True


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if "district" in frame.columns:
        frame["latitude"] = frame["district"].map(lambda d: DISTRICTS.get(d, (np.nan, np.nan))[0])
        frame["longitude"] = frame["district"].map(lambda d: DISTRICTS.get(d, (np.nan, np.nan))[1])
    if "health_impact_risk_level" in frame.columns:
        frame["health_impact_risk_level"] = (
            frame["health_impact_risk_level"].astype(str).str.upper().replace({"NAN": "UNKNOWN"})
        )
    if "thermal_risk_level" in frame.columns:
        frame["thermal_risk_level"] = frame["thermal_risk_level"].astype(str)

    # Transparent "Heatwave Chance" proxy: Yes if the district is on an
    # extreme-heat streak or already past its 95th-percentile WBGT threshold.
    if "hot_day_streak" in frame.columns and "wbgt_above_p95" in frame.columns:
        chance = (
            (pd.to_numeric(frame["hot_day_streak"], errors="coerce").fillna(0) >= 2)
            | (pd.to_numeric(frame["wbgt_above_p95"], errors="coerce").fillna(0) == 1)
        )
        frame["heatwave_chance"] = np.where(chance, "Yes", "No")
    elif "thermal_risk_level" in frame.columns:
        frame["heatwave_chance"] = np.where(
            frame["thermal_risk_level"].isin(["Very High", "Extreme"]), "Yes", "No"
        )
    else:
        frame["heatwave_chance"] = "—"
    return frame


def badge(title: str, label, color_map: dict) -> None:
    label_str = "—" if label is None or (isinstance(label, float) and pd.isna(label)) else str(label)
    color = color_map.get(label_str, "#95a5a6")
    st.markdown(
        f"""
        <div style="margin-bottom:0.75rem;">
          <div style="font-size:0.8rem;color:#8a8a8a;margin-bottom:2px;">{title}</div>
          <div style="display:inline-block;padding:4px 16px;border-radius:14px;
                      background:{color};color:white;font-weight:600;font-size:0.95rem;">
            {label_str}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Plotly renamed Scattermapbox -> Scattermap (and the layout key
# mapbox -> map) in newer releases. Support both so the dashboard works
# regardless of which Plotly version is installed.
if hasattr(go, "Scattermap"):
    _ScatterMapTrace = go.Scattermap
    _MAP_LAYOUT_KEY = "map"
else:
    _ScatterMapTrace = go.Scattermapbox
    _MAP_LAYOUT_KEY = "mapbox"


def build_risk_map(map_df: pd.DataFrame) -> go.Figure:
    """Color-coded GIS risk map — RED / ORANGE / YELLOW / GREEN risk zones."""
    fig = go.Figure()
    score_col = "health_impact_risk_score" if "health_impact_risk_score" in map_df.columns else None

    for level in RISK_ORDER:
        sub = map_df[map_df["health_impact_risk_level"] == level]
        if sub.empty:
            continue
        color = RISK_COLORS[level]
        sizes = (
            np.clip(pd.to_numeric(sub[score_col], errors="coerce").fillna(30), 15, 100)
            if score_col else pd.Series(35, index=sub.index)
        )
        hover_text = sub.apply(
            lambda r: (
                f"<b>{r['district']}</b><br>"
                f"Risk level: {level}<br>"
                + (f"Risk score: {r[score_col]:.1f}<br>" if score_col and pd.notna(r[score_col]) else "")
                + (f"HTSI: {r['human_thermal_stress_index']:.1f}<br>" if "human_thermal_stress_index" in r and pd.notna(r["human_thermal_stress_index"]) else "")
                + (f"Heatwave chance: {r['heatwave_chance']}" if "heatwave_chance" in r else "")
            ),
            axis=1,
        )
        # Large translucent circle = risk "zone" shading, echoing the sketch
        fig.add_trace(_ScatterMapTrace(
            lat=sub["latitude"], lon=sub["longitude"],
            mode="markers",
            marker=dict(size=sizes * 1.8, color=color, opacity=0.30),
            hoverinfo="skip",
            showlegend=False,
        ))
        # Solid center marker + legend entry
        fig.add_trace(_ScatterMapTrace(
            lat=sub["latitude"], lon=sub["longitude"],
            mode="markers+text",
            marker=dict(size=14, color=color, opacity=0.95),
            text=sub["district"],
            textposition="top center",
            textfont=dict(size=10, color="#333"),
            name=f"{level} ({len(sub)})",
            hovertext=hover_text,
            hoverinfo="text",
        ))

    fig.update_layout(
        **{_MAP_LAYOUT_KEY: dict(style="carto-positron", center=dict(lat=26.3, lon=92.9), zoom=5.6)},
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(255,255,255,0.6)"),
    )
    return fig


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
st.title("🌡️ Assam Heat-Health Early Warning Dashboard")
st.caption("Weather → Human Thermal Stress → Vulnerability → Health Risk → Forecast → GIS → Alerts")

# The dashboard refreshes its live forecast cache every 5 minutes and also
# provides an explicit button for an immediate latest-data fetch.
refresh_col, status_col = st.columns([1, 4])
with refresh_col:
    if st.button("🔄 Refresh latest prediction", use_container_width=True):
        load_risk_data.clear()
        st.rerun()
with status_col:
    st.caption("Live forecast source: Open-Meteo • automatic refresh window: 5 minutes • click refresh for an immediate update")

df, source, is_live_computed = load_risk_data(0)

if df.empty:
    st.error("The latest weather forecast could not be fetched. Check your internet connection and try the refresh button.")
    st.stop()

st.caption(f"🟢 {source} — risk calculated from the newest fetched forecast")

# --------------------------------------------------------------------------
# Top filter bar — Location | Date | MAP | Category
# --------------------------------------------------------------------------
f_loc, f_date, f_title, f_cat = st.columns([1.2, 1.2, 1.6, 1.2])

with f_loc:
    districts_sorted = sorted(df["district"].dropna().unique())
    location = st.selectbox("📍 Location", ["All Districts"] + districts_sorted)

with f_date:
    dates_sorted = sorted(df["date"].dropna().unique())
    date_sel = st.selectbox(
        "📅 Date", dates_sorted,
        format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"),
    )

with f_title:
    st.markdown(
        "<h2 style='text-align:center;margin-top:0.35rem;color:#444;'>MAP</h2>",
        unsafe_allow_html=True,
    )

with f_cat:
    category = st.selectbox("🎯 Risk Category", ["All", "RED", "ORANGE", "YELLOW", "GREEN"])

# --------------------------------------------------------------------------
# Apply filters
# --------------------------------------------------------------------------
view = df[df["date"] == date_sel].copy()
if location != "All Districts":
    view = view[view["district"] == location]
if category != "All":
    view = view[view["health_impact_risk_level"] == category]

# --------------------------------------------------------------------------
# Statewide KPI strip for the selected date
# --------------------------------------------------------------------------
day_df = df[df["date"] == date_sel]
k0, k1, k2, k3, k4 = st.columns(5)
k0.metric("Districts reporting", day_df["district"].nunique())
for col, level in zip((k1, k2, k3, k4), ("RED", "ORANGE", "YELLOW", "GREEN")):
    count = int((day_df["health_impact_risk_level"] == level).sum())
    col.metric(f"{level.title()} districts", count)

st.divider()

# --------------------------------------------------------------------------
# Main layout — left: district snapshot, right: color-coded risk map
# --------------------------------------------------------------------------
left, right = st.columns([1, 1.7], gap="large")

with left:
    st.markdown("#### District Snapshot")

    if view.empty:
        st.info("No data matches the current filters.")
        focus_row = None
    elif location == "All Districts":
        focus_row = view.sort_values(
            "health_impact_risk_score", ascending=False
        ).iloc[0] if "health_impact_risk_score" in view.columns else view.iloc[0]
        st.caption(f"Highest-risk district shown: **{focus_row['district']}**")
    else:
        focus_row = view.iloc[0]

    if focus_row is not None:
        st.metric("🔥 Heatwave Chance", focus_row.get("heatwave_chance", "—"))

        htsi = focus_row.get("human_thermal_stress_index", np.nan)
        st.metric("🌡️ Human Thermal Stress Index", f"{htsi:.1f}" if pd.notna(htsi) else "—")

        badge("Thermal Risk Level", focus_row.get("thermal_risk_level"), THERMAL_COLORS)

        score = focus_row.get("health_impact_risk_score", np.nan)
        st.metric("🏥 Health Impact Score", f"{score:.1f}" if pd.notna(score) else "—")

        badge("Health Impact Level", focus_row.get("health_impact_risk_level"), RISK_COLORS)

        if "advisory_actions" in focus_row.index and pd.notna(focus_row["advisory_actions"]):
            with st.expander("📋 Recommended actions"):
                for action in str(focus_row["advisory_actions"]).split("|"):
                    action = action.strip()
                    if action:
                        st.write(f"- {action}")

with right:
    map_df = view.dropna(subset=["latitude", "longitude"])
    if map_df.empty:
        st.info("No mapped districts for the current filter combination.")
    else:
        st.plotly_chart(build_risk_map(map_df), use_container_width=True)

    m_loc, m_time, m_alert = st.columns(3)
    with m_loc:
        st.caption("📍 Location")
        st.write(location)
    with m_time:
        st.caption("🕒 Date")
        st.write(pd.Timestamp(date_sel).strftime("%d %b %Y"))
    with m_alert:
        st.caption("🚨 Alert")
        alert_count = int(view["health_impact_risk_level"].isin(["RED", "ORANGE"]).sum())
        if st.button(f"View Alerts ({alert_count})", use_container_width=True):
            st.session_state["show_alerts"] = True

st.divider()

# --------------------------------------------------------------------------
# Filtered data table
# --------------------------------------------------------------------------
st.subheader("📊 Filtered Risk Table")
show_cols = [c for c in [
    "district", "date", "heatwave_chance", "human_thermal_stress_index",
    "thermal_risk_level", "health_impact_risk_score", "health_impact_risk_level",
    "advisory_actions",
] if c in view.columns]
sort_col = "health_impact_risk_score" if "health_impact_risk_score" in view.columns else show_cols[0]
st.dataframe(view[show_cols].sort_values(sort_col, ascending=False), use_container_width=True, height=360)

# --------------------------------------------------------------------------
# Alert feed
# --------------------------------------------------------------------------
st.subheader("🚨 Alert Feed")
alert_rows = view[view["health_impact_risk_level"].isin(["RED", "ORANGE"])] if "health_impact_risk_level" in view.columns else pd.DataFrame()
if alert_rows.empty:
    st.success("No RED/ORANGE heat-health alerts for the current filter.")
else:
    for _, r in alert_rows.sort_values("health_impact_risk_score", ascending=False).iterrows():
        level = r["health_impact_risk_level"]
        score_txt = f" (score {r['health_impact_risk_score']:.1f})" if pd.notna(r.get("health_impact_risk_score")) else ""
        msg = f"**{r['district']}** — {level} heat-health risk{score_txt}."
        if pd.notna(r.get("advisory_actions")):
            first_action = str(r["advisory_actions"]).split("|")[0].strip()
            msg += f" Recommended: {first_action}."
        if level == "RED":
            st.error(msg)
        else:
            st.warning(msg)