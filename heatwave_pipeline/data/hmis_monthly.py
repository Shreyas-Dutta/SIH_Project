"""Monthly HMIS health-vulnerability engine.

The V10 project previously treated HMIS as a static district baseline.  This
module supports the preferred structure: indicator observations keyed by
sub-district/facility and reporting month, aggregated to district-month and
then joined to daily weather on district + year + month.

It deliberately refuses to invent month-to-month HMIS values when only a
static district snapshot is available.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from ..config import BASE_DIR, DISTRICTS
from ..utils.helpers import clean_text, safe_numeric, save_csv, log

ALIASES = {
    "kamrup metropolitan": "Kamrup M", "kamrup metro": "Kamrup M", "kamrup m": "Kamrup M",
    "kamrup rural": "Kamrup R", "kamrup r": "Kamrup R", "morigaon": "Marigaon",
    "marigaon": "Marigaon", "sibsagar": "Sibsagar", "sivasagar": "Sibsagar",
    "dima hasao": "Dima Hasao", "dimahasao": "Dima Hasao", "north cachar hills": "Dima Hasao",
}

INDICATOR_GROUPS = {
    "respiratory": ["asthma", "copd", "chronic obstructive", "respiratory"],
    "vector_borne": ["dengue", "malaria", "pyrexia", "vector"],
    "gastrointestinal": ["diarrhea", "diarrhoea", "dehydration", "gastro"],
    "infectious": ["typhoid", "hepatitis", "tuberculosis", "tb", "fever"],
    "mortality": ["death", "deaths", "mortality"],
    "severe_care": ["emergency", "sncu", "nrc", "inpatient", "admission", "admitted"],
    "maternal_child": ["maternal", "infant", "newborn", "new born", "child"],
}

WEIGHTS = {
    "respiratory": 0.18,
    "vector_borne": 0.15,
    "gastrointestinal": 0.15,
    "infectious": 0.12,
    "mortality": 0.18,
    "severe_care": 0.12,
    "maternal_child": 0.10,
}


def _canonical_district(x):
    s = re.sub(r"\s+", " ", clean_text(x).lower()).strip()
    if s in ALIASES:
        return ALIASES[s]
    for d in DISTRICTS:
        if s == d.lower():
            return d
    return None


def _find_col(df, names):
    low = {clean_text(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    for c in df.columns:
        lc = clean_text(c).lower()
        if any(n.lower() in lc for n in names):
            return c
    return None


def _parse_month(df):
    date_col = _find_col(df, ["date", "month", "reporting_month", "report_month", "period", "year_month"])
    year_col = _find_col(df, ["year", "reporting_year", "report_year"])
    month_col = _find_col(df, ["month_no", "month_number", "month_num"])

    out = df.copy()
    if date_col:
        raw = out[date_col].astype(str).str.strip()
        dt = pd.to_datetime(raw, errors="coerce", dayfirst=False)
        if dt.notna().mean() < 0.5:
            # Handles strings such as Apr-2020 / 2020-Apr.
            dt = pd.to_datetime(raw, format="%b-%Y", errors="coerce")
        out["report_month"] = dt.dt.to_period("M").astype(str)
    elif year_col and month_col:
        y = pd.to_numeric(out[year_col], errors="coerce")
        m = pd.to_numeric(out[month_col], errors="coerce")
        out["report_month"] = pd.to_datetime(
            dict(year=y, month=m, day=1), errors="coerce"
        ).dt.to_period("M").astype(str)
    else:
        out["report_month"] = pd.NA
    return out


def _choose_value_column(df):
    candidates = [
        "value", "total", "count", "reported_value", "indicator_value",
        "number", "observation", "value_total", "total_value",
    ]
    c = _find_col(df, candidates)
    if c:
        return c
    # Prefer numeric columns while excluding coordinates/IDs.
    best = None
    best_score = 0
    for c in df.columns:
        lc = clean_text(c).lower()
        if any(k in lc for k in ["lat", "lon", "code", "year", "month"]):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        score = s.notna().mean()
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.5 else None


def discover_monthly_files():
    roots = [BASE_DIR / "hmis_monthly", BASE_DIR / "data" / "hmis_monthly"]
    files = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.glob("*.csv")))
    return files


def load_monthly_hmis():
    """Load monthly/sub-district HMIS files if they are actually present."""
    files = discover_monthly_files()
    if not files:
        return None

    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
            if df.empty:
                continue
            frames.append(df)
        except Exception as exc:
            log(f"[WARNING] Could not read monthly HMIS file {path.name}: {exc}")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def build_monthly_vulnerability(raw):
    """Return district-month vulnerability from actual monthly HMIS observations."""
    if raw is None or raw.empty:
        return None

    df = _parse_month(raw)
    district_col = _find_col(df, ["district", "district_name", "district name"])
    if district_col is None:
        return None
    value_col = _choose_value_column(df)
    indicator_col = _find_col(df, ["indicator", "indicator_name", "health_indicator", "parameter", "parameters", "item"])
    if value_col is None or indicator_col is None or "report_month" not in df:
        return None

    df["district"] = df[district_col].map(_canonical_district)
    df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    df["indicator_text"] = df[indicator_col].fillna("").astype(str).str.lower()
    df = df.dropna(subset=["district", "report_month", "value"]).copy()
    if df.empty:
        return None

    # Keep only health-burden indicators represented by the supplied HMIS keys.
    parts = []
    for group, terms in INDICATOR_GROUPS.items():
        mask = pd.Series(False, index=df.index)
        for term in terms:
            mask |= df["indicator_text"].str.contains(term, regex=False, na=False)
        x = df.loc[mask, ["district", "report_month", "value"]].copy()
        if x.empty:
            continue
        x["group"] = group
        parts.append(x.groupby(["district", "report_month", "group"], as_index=False)["value"].sum())

    if not parts:
        return None
    long = pd.concat(parts, ignore_index=True)

    # Normalize each group across district-month observations. Log damping
    # reduces domination by large districts and percentile scaling preserves
    # relative monthly burden.
    component_frames = []
    for group, g in long.groupby("group"):
        g = g.copy()
        z = np.log1p(g["value"].clip(lower=0))
        g["component"] = z.rank(method="average", pct=True) * 100.0
        component_frames.append(g[["district", "report_month", "group", "component"]])
    comp = pd.concat(component_frames, ignore_index=True)

    comp["weight"] = comp["group"].map(WEIGHTS).fillna(0.0)
    scored = comp.groupby(["district", "report_month"], as_index=False).apply(
        lambda x: pd.Series({
            "health_vulnerability_score": np.average(x["component"], weights=x["weight"]) if x["weight"].sum() else 50.0,
            "health_hmis_component_count": int(x["group"].nunique()),
        }), include_groups=False
    ).reset_index(drop=True)

    scored["health_vulnerability_score"] = scored["health_vulnerability_score"].clip(0, 100).round(1)
    scored["health_vulnerability_level"] = pd.cut(
        scored["health_vulnerability_score"],
        [-np.inf, 20, 40, 60, 80, np.inf],
        labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"],
        include_lowest=True,
    ).astype(str)
    scored["vulnerability_data_status"] = "HMIS_MONTHLY_SUBDISTRICT_AGGREGATED"
    scored["vulnerability_components"] = scored["health_hmis_component_count"].astype(str) + " HMIS groups"
    return scored


def attach_monthly_vulnerability(weather, monthly):
    out = weather.copy()
    if monthly is None or monthly.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["report_month"] = out["date"].dt.to_period("M").astype(str)
    out["district"] = out["district"].map(_canonical_district).fillna(out["district"])
    m = monthly.copy()
    m["district"] = m["district"].map(_canonical_district).fillna(m["district"])
    out = out.drop(columns=[c for c in ["health_vulnerability_score", "health_vulnerability_level", "vulnerability_data_status", "vulnerability_components"] if c in out], errors="ignore")
    out = out.merge(m, on=["district", "report_month"], how="left", validate="many_to_one")
    out = out.drop(columns=["report_month"], errors="ignore")
    return out
