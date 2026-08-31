"""Assam Heat-Health Early Warning Dashboard — polished UI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heatwave_pipeline.config import DISTRICTS
from heatwave_pipeline.data.weather import fetch_forecast
from heatwave_pipeline.dataset.builder import standardize_weather_columns
from heatwave_pipeline.features.health_risk import add_health_risk_features
from heatwave_pipeline.features.thermal import add_thermal_features
from heatwave_pipeline.risk.scorer import score_health_impact
from heatwave_pipeline.forecast.risk_forecast import add_forecast_heatwave_probability

st.set_page_config(
    page_title="Assam Heat-Health Early Warning",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme / UI
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "GREEN": "#16a34a",
    "YELLOW": "#eab308",
    "ORANGE": "#f97316",
    "RED": "#dc2626",
    "UNKNOWN": "#64748b",
}
RISK_ORDER = ["RED", "ORANGE", "YELLOW", "GREEN", "UNKNOWN"]
THERMAL_COLORS = {
    "Low": "#16a34a",
    "Moderate": "#eab308",
    "High": "#f97316",
    "Very High": "#dc2626",
    "Extreme": "#7c3aed",
    "nan": "#64748b",
}

st.markdown(
    """
<style>
:root { color-scheme: light !important; }
html, body, .stApp { background:#f4f7fb !important; color:#172033 !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#f4f7fb !important; }
[data-testid="stHeader"] { background:rgba(244,247,251,.97) !important; }
.block-container { padding-top:1.5rem; padding-bottom:2.5rem; max-width:1500px; }

/* Main text: explicit high contrast */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 { color:#0f172a !important; }
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li { color:#334155 !important; }
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] { color:#64748b !important; }

/* Dark sidebar */
[data-testid="stSidebar"] { background:#0b1324 !important; border-right:1px solid #1e293b !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] label { color:#dbe5f1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 { color:#f8fafc !important; }
[data-testid="stSidebar"] [data-baseweb="select"] { background:#111827 !important; border-color:#334155 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#f8fafc !important; }
[data-testid="stSidebar"] .stButton > button { color:#f8fafc !important; background:#202b3f !important; border:1px solid #475569 !important; }

/* Hero */
.hero { background:linear-gradient(135deg,#0f172a 0%,#172554 55%,#0f766e 100%); border-radius:22px; padding:28px 32px; color:#fff !important; margin-bottom:18px; box-shadow:0 12px 35px rgba(15,23,42,.18); }
.hero h1 { margin:0; font-size:2.05rem; letter-spacing:-.03em; color:#fff !important; }
.hero p { margin:.45rem 0 0; color:#dbeafe !important; font-size:.98rem; }
.live-pill { display:inline-flex; align-items:center; gap:7px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.20); border-radius:999px; padding:6px 11px; font-size:.78rem; margin-bottom:12px; color:#fff !important; }
.live-dot { width:8px; height:8px; border-radius:50%; background:#22c55e; box-shadow:0 0 0 4px rgba(34,197,94,.16); }

/* Cards */
.card, .status-card { background:#fff !important; border:1px solid #d5dee9 !important; border-radius:16px; padding:17px 18px; box-shadow:0 4px 18px rgba(15,23,42,.06); height:100%; box-sizing:border-box; }
.card-label, .status-title { color:#475569 !important; font-size:.76rem; font-weight:800; text-transform:uppercase; letter-spacing:.06em; }
.card-value, .status-value { color:#0f172a !important; font-size:1.65rem; font-weight:800; margin-top:5px; }
.card-sub { color:#64748b !important; font-size:.76rem; margin-top:3px; }
.section-title { color:#0f172a !important; font-size:1.15rem; font-weight:800; margin:4px 0 10px; }
.risk-chip { display:inline-block; border-radius:999px; padding:6px 13px; color:#fff !important; font-size:.78rem; font-weight:800; line-height:1.1; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] { gap:8px; border-bottom:1px solid #d5dee9; }
[data-testid="stTabs"] button[role="tab"] { color:#334155 !important; background:#fff !important; border:1px solid #d5dee9 !important; border-radius:10px 10px 0 0; padding:9px 16px; font-weight:800 !important; }
[data-testid="stTabs"] button[role="tab"] *,
[data-testid="stTabs"] button[role="tab"] p { color:#334155 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#b91c1c !important; background:#fff !important; border-bottom:3px solid #dc2626 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color:#b91c1c !important; }

/* Expanders */
[data-testid="stExpander"] { background:#fff !important; border:1px solid #d5dee9 !important; border-radius:12px !important; overflow:hidden; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color:#0f172a !important; font-weight:800 !important; }
[data-testid="stExpanderDetails"],
[data-testid="stExpanderDetails"] p,
[data-testid="stExpanderDetails"] li,
[data-testid="stExpanderDetails"] span { color:#334155 !important; }

/* Alerts and native controls */
.alert-item { background:#fff !important; border:1px solid #d5dee9; border-left:5px solid #f97316; border-radius:12px; padding:13px 16px; margin:8px 0; box-shadow:0 3px 12px rgba(15,23,42,.04); color:#172033 !important; }
.alert-item b { color:#0f172a !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color:#1e293b !important; }
[data-testid="stAppViewContainer"] [data-baseweb="select"] { background:#fff !important; }
[data-testid="stAppViewContainer"] [data-baseweb="select"] * { color:#172033 !important; }
[data-testid="stAppViewContainer"] input, [data-testid="stAppViewContainer"] textarea { color:#172033 !important; background:#fff !important; }
.stButton > button, .stDownloadButton > button { border-radius:10px !important; font-weight:800 !important; }
div[data-testid="stMetric"] { background:#fff !important; border:1px solid #d5dee9 !important; padding:13px 15px; border-radius:14px; box-shadow:0 4px 18px rgba(15,23,42,.04); }
[data-testid="stDataFrame"] { background:#fff !important; border:1px solid #d5dee9 !important; border-radius:12px; overflow:hidden; }
[data-testid="stPlotlyChart"] { background:#fff !important; border:1px solid #d5dee9; border-radius:14px; padding:4px; }
.footer { text-align:center; color:#64748b !important; font-size:.75rem; padding:18px 0 0; }
@media (max-width:900px) { .hero { padding:22px; } .hero h1 { font-size:1.55rem; } }
</style>
""",
    unsafe_allow_html=True,
)


def badge(title: str, label, color_map: dict) -> None:
    label_str = "—" if label is None or (isinstance(label, float) and pd.isna(label)) else str(label)
    color = color_map.get(label_str, "#64748b")
    st.markdown(
        f'<div class="card"><div class="card-label">{title}</div><div style="margin-top:8px"><span class="risk-chip" style="background:{color}">{label_str}</span></div></div>',
        unsafe_allow_html=True,
    )


def card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300, show_spinner="Fetching the latest weather forecast…")
def load_risk_data(refresh_token: int = 0):
    del refresh_token
    forecast = fetch_forecast(force_refresh=True)
    if forecast is None or forecast.empty:
        return pd.DataFrame(), "Latest Open-Meteo fetch failed", False

    forecast = standardize_weather_columns(forecast)
    forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
    forecast = forecast.dropna(subset=["date", "district"]).copy()

    baseline_path = ROOT / "hmis_district_baseline.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None
    computed = add_thermal_features(forecast)
    computed = add_forecast_heatwave_probability(computed)
    computed = add_health_risk_features(computed, baseline)
    computed = score_health_impact(computed, heat_probability_col="heatwave_probability_pct")

    keep = [c for c in [
        "date", "district", "human_thermal_stress_index", "thermal_risk_level",
        "heatwave_probability_pct", "heatwave_forecast_event", "hot_day_streak", "wbgt_above_p95", "health_vulnerability_score",
        "health_impact_risk_score", "health_impact_risk_level", "advisory_actions",
        "fetched_at_ist",
    ] if c in computed.columns]
    computed = computed[keep].sort_values(["date", "health_impact_risk_score"], ascending=[True, False])
    fetched_at = (
        str(computed["fetched_at_ist"].dropna().iloc[0])
        if "fetched_at_ist" in computed.columns and computed["fetched_at_ist"].notna().any()
        else pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    )
    return _finalize(computed), f"Live Open-Meteo forecast • fetched {fetched_at}", True


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["latitude"] = frame["district"].map(lambda d: DISTRICTS.get(d, (np.nan, np.nan))[0])
    frame["longitude"] = frame["district"].map(lambda d: DISTRICTS.get(d, (np.nan, np.nan))[1])
    frame["health_impact_risk_level"] = frame["health_impact_risk_level"].astype(str).str.upper().replace({"NAN": "UNKNOWN"})
    frame["thermal_risk_level"] = frame["thermal_risk_level"].astype(str)
    prob = pd.to_numeric(frame.get("heatwave_probability_pct", np.nan), errors="coerce")
    event = frame.get("heatwave_forecast_event", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame["heatwave_chance"] = prob.round(0).astype("Int64").astype(str).replace("<NA>", "—") + "%"
    frame["heatwave_event_label"] = np.where(event, "Likely event (2+ of next 3 days)", "No 2-of-3 heatwave signal")
    return frame


if hasattr(go, "Scattermap"):
    _ScatterMapTrace = go.Scattermap
    _MAP_LAYOUT_KEY = "map"
else:
    _ScatterMapTrace = go.Scattermapbox
    _MAP_LAYOUT_KEY = "mapbox"


def build_risk_map(map_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    score_col = "health_impact_risk_score"
    for level in RISK_ORDER:
        sub = map_df[map_df["health_impact_risk_level"] == level]
        if sub.empty:
            continue
        color = RISK_COLORS[level]
        sizes = np.clip(pd.to_numeric(sub[score_col], errors="coerce").fillna(30), 18, 90)
        hover = sub.apply(
            lambda r: f"<b>{r['district']}</b><br>Risk: {level}<br>Score: {r[score_col]:.1f}<br>HTSI: {r['human_thermal_stress_index']:.1f}<br>Heatwave likelihood: {r['heatwave_chance']}<br>{r['heatwave_event_label']}",
            axis=1,
        )
        fig.add_trace(_ScatterMapTrace(lat=sub["latitude"], lon=sub["longitude"], mode="markers", marker=dict(size=sizes*1.9, color=color, opacity=.18), hoverinfo="skip", showlegend=False))
        fig.add_trace(_ScatterMapTrace(lat=sub["latitude"], lon=sub["longitude"], mode="markers+text", marker=dict(size=13, color=color, opacity=.95), text=sub["district"], textposition="top center", textfont=dict(size=9, color="#334155"), name=f"{level.title()} ({len(sub)})", hovertext=hover, hoverinfo="text"))
    fig.update_layout(
        **{_MAP_LAYOUT_KEY: dict(style="carto-positron", center=dict(lat=26.3, lon=92.9), zoom=5.55)},
        margin=dict(l=0, r=0, t=0, b=0), height=585,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(255,255,255,.75)", font=dict(size=11)),
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><div class="live-pill"><span class="live-dot"></span> LIVE EARLY WARNING SYSTEM</div><h1>Assam Heat-Health Risk Monitor</h1><p>Real-time weather intelligence transformed into thermal stress, vulnerability and district-level health risk.</p></div>',
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.markdown("## 🌡️ Control Center")
    st.caption("Choose what you want to monitor. The system uses the newest forecast available from Open-Meteo.")
    if st.button("🔄 Refresh latest prediction", width="stretch"):
        load_risk_data.clear()
        st.rerun()
    st.markdown("---")

df, source, is_live = load_risk_data(0)
if df.empty:
    st.error("The latest weather forecast could not be fetched. Check your internet connection and try Refresh.")
    st.stop()

with st.sidebar:
    districts_sorted = sorted(df["district"].dropna().unique())
    location = st.selectbox("📍 District", ["All Districts"] + districts_sorted)
    dates_sorted = sorted(df["date"].dropna().unique())
    date_sel = st.selectbox("📅 Forecast date", dates_sorted, format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"))
    category = st.selectbox("🎯 Risk category", ["All", "RED", "ORANGE", "YELLOW", "GREEN"])
    st.markdown("---")
    st.caption("DATA PIPELINE")
    st.markdown("**Weather:** Open-Meteo  ")
    st.markdown("**Refresh:** Every 5 minutes  ")
    st.markdown("**Risk:** Live recalculation")

# Main-content wrapper gives CSS a stable scope and prevents theme-dependent
# text colors from making dashboard elements disappear.
view = df[df["date"] == date_sel].copy()
if location != "All Districts":
    view = view[view["district"] == location]
if category != "All":
    view = view[view["health_impact_risk_level"] == category]
day_df = df[df["date"] == date_sel].copy()

# Status bar
status_col, refresh_col = st.columns([5, 1])
with status_col:
    st.markdown(f'<div class="status-card"><span class="status-title">Data status</span><div class="status-value">🟢 {source}</div></div>', unsafe_allow_html=True)
with refresh_col:
    st.markdown(f'<div class="status-card"><span class="status-title">Selected date</span><div class="status-value">{pd.Timestamp(date_sel).strftime("%d %b")}</div></div>', unsafe_allow_html=True)

# KPI row
st.markdown("<div class='section-title'>Statewide risk overview</div>", unsafe_allow_html=True)
kpis = st.columns(5)
with kpis[0]: card("Districts reporting", str(day_df["district"].nunique()), "latest forecast")
for i, level in enumerate(["RED", "ORANGE", "YELLOW", "GREEN"], start=1):
    with kpis[i]:
        n = int((day_df["health_impact_risk_level"] == level).sum())
        card(f"{level.title()} risk", str(n), "districts")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
tab_overview, tab_forecast, tab_alerts, tab_data = st.tabs(["🗺️ Overview", "📈 Forecast", "🚨 Alerts", "📊 Data"])

with tab_overview:
    left, right = st.columns([0.9, 1.7], gap="large")
    with left:
        st.markdown("<div class='section-title'>District snapshot</div>", unsafe_allow_html=True)
        if view.empty:
            st.info("No data matches the selected filters.")
        else:
            focus = view.sort_values("health_impact_risk_score", ascending=False).iloc[0]
            if location == "All Districts":
                st.caption(f"Highest risk in current view: **{focus['district']}**")
            else:
                st.caption(f"Monitoring **{focus['district']}**")
            a, b = st.columns(2)
            with a: card("Heatwave chance", str(focus.get("heatwave_chance", "—")), "model likelihood • next 3 days")
            with b:
                htsi = focus.get("human_thermal_stress_index", np.nan)
                card("Thermal stress", f"{htsi:.1f}" if pd.notna(htsi) else "—", "HTSI")
            event_label = str(focus.get("heatwave_event_label", ""))
            if event_label and event_label != "nan":
                st.caption(f"Heatwave signal: **{event_label}**")
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                badge("Thermal risk level", focus.get("thermal_risk_level"), THERMAL_COLORS)
            with b:
                score = focus.get("health_impact_risk_score", np.nan)
                badge("Health impact level", focus.get("health_impact_risk_level"), RISK_COLORS)
                if pd.notna(score):
                    st.markdown(f'<div class="card-sub" style="margin-top:-28px;margin-left:18px">Score: <b>{score:.1f}</b></div>', unsafe_allow_html=True)
            actions = focus.get("advisory_actions")
            if pd.notna(actions):
                with st.expander("📋 Recommended protective actions", expanded=True):
                    for action in str(actions).split("|"):
                        action = action.strip()
                        if action: st.markdown(f"• {action}")
    with right:
        st.markdown("<div class='section-title'>District risk map</div>", unsafe_allow_html=True)
        map_df = view.dropna(subset=["latitude", "longitude"])
        if map_df.empty:
            st.info("No mapped districts for the current filters.")
        else:
            st.plotly_chart(build_risk_map(map_df), width="stretch", config={"displayModeBar": False})

with tab_forecast:
    st.markdown("<div class='section-title'>5-day health-risk trajectory</div>", unsafe_allow_html=True)
    chart_df = df.copy()
    if location != "All Districts": chart_df = chart_df[chart_df["district"] == location]
    if category != "All": chart_df = chart_df[chart_df["health_impact_risk_level"] == category]
    if not chart_df.empty:
        daily = chart_df.groupby("date", as_index=False)["health_impact_risk_score"].mean()
        fig = px.line(daily, x="date", y="health_impact_risk_score", markers=True, labels={"date":"Date", "health_impact_risk_score":"Average health-impact score"})
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=15,b=0), paper_bgcolor="white", plot_bgcolor="white", yaxis=dict(range=[0,100], gridcolor="#e2e8f0"), xaxis=dict(gridcolor="#f1f5f9"))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        pivot = chart_df.pivot_table(index="date", columns="health_impact_risk_level", values="district", aggfunc="count", fill_value=0).reset_index()
        for c in RISK_ORDER:
            if c not in pivot.columns: pivot[c] = 0
        melt = pivot.melt(id_vars="date", value_vars=["GREEN","YELLOW","ORANGE","RED"], var_name="Risk", value_name="Districts")
        fig2 = px.bar(melt, x="date", y="Districts", color="Risk", barmode="stack", color_discrete_map=RISK_COLORS)
        fig2.update_layout(height=330, margin=dict(l=0,r=0,t=15,b=0), paper_bgcolor="white", plot_bgcolor="white", legend_title_text="")
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

with tab_alerts:
    alert_rows = view[view["health_impact_risk_level"].isin(["RED", "ORANGE"])].sort_values("health_impact_risk_score", ascending=False)
    st.markdown(f"<div class='section-title'>Active alerts <span style='color:#64748b;font-size:.85rem'>({len(alert_rows)})</span></div>", unsafe_allow_html=True)
    if alert_rows.empty:
        st.success("No RED or ORANGE heat-health alerts for the current selection.")
    else:
        for _, r in alert_rows.iterrows():
            level = r["health_impact_risk_level"]
            score = r.get("health_impact_risk_score")
            action = str(r.get("advisory_actions", "")).split("|")[0].strip()
            border = RISK_COLORS.get(level, "#f97316")
            st.markdown(f'<div class="alert-item" style="border-left-color:{border}"><b>{r["district"]}</b> · <span style="color:{border};font-weight:800">{level}</span> · score {score:.1f}<br><span style="color:#64748b">{action}</span></div>', unsafe_allow_html=True)

with tab_data:
    st.markdown("<div class='section-title'>Filtered risk table</div>", unsafe_allow_html=True)
    show_cols = [c for c in ["district","date","heatwave_chance","heatwave_event_label","human_thermal_stress_index","thermal_risk_level","health_impact_risk_score","health_impact_risk_level","advisory_actions"] if c in view.columns]
    table = view[show_cols].sort_values("health_impact_risk_score", ascending=False) if "health_impact_risk_score" in view.columns else view[show_cols]
    st.dataframe(table, width="stretch", height=460, hide_index=True)
    st.download_button("⬇️ Export current view as CSV", table.to_csv(index=False).encode("utf-8"), "assam_heat_health_risk.csv", "text/csv", width="content")

st.markdown('<div class="footer">Assam Heat-Health Early Warning System · Live forecast refresh every 5 minutes · Risk is recalculated from the newest fetched weather data</div>', unsafe_allow_html=True)
