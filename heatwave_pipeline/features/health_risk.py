"""Health-vulnerability features from the available HMIS baseline.

The HMIS source is a district-level snapshot of service/outcome counts, not a
population-normalised epidemiological survey.  Therefore this module produces
an explicitly labelled *relative baseline vulnerability proxy* rather than a
medical probability.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _district_percentile(s: pd.Series) -> pd.Series:
    """Robust 0..1 cross-district percentile for a non-negative burden."""
    x = pd.to_numeric(s, errors="coerce")
    valid = x.notna()
    if valid.sum() <= 1:
        return pd.Series(0.5, index=s.index, dtype=float)
    # Log dampens the effect of very large districts/service volumes.
    x = np.log1p(x.clip(lower=0))
    ranks = x[valid].rank(method="average", pct=True)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[valid] = ranks
    # Median imputation only for missing indicators; missing is not interpreted as zero.
    return out.fillna(0.5).clip(0, 1)


def _matching_columns(columns, include_terms, require_total=True):
    found = []
    for c in columns:
        low = str(c).lower()
        # HMIS usually exposes a *_total field, but some indicators are already
        # district totals and therefore have no category suffix (e.g. the
        # asthma/COPD indicator). Keep those unsuffixed fields.
        if require_total and not ("_total" in low or low.endswith("_copd") or low.endswith("_asthma")):
            continue
        if any(term in low for term in include_terms):
            found.append(c)
    return found


def build_district_vulnerability(baseline):
    """Create one vulnerability score per district from heat-relevant HMIS burdens.

    The score is intentionally based on *heat-sensitive health burdens* rather
    than every numeric HMIS field.  General admission counts, private/public
    splits and unrelated diseases are excluded because they can measure service
    utilisation rather than susceptibility.
    """
    cols = ["district", "health_vulnerability_score", "health_vulnerability_level",
            "vulnerability_data_status", "vulnerability_components"]
    if baseline is None or baseline.empty or "district" not in baseline.columns:
        return pd.DataFrame(columns=cols)

    out = baseline.copy()
    groups = {
        "respiratory": _matching_columns(out.columns, ["asthma", "chronic_obstructive_pulmonary_disease", "copd", "respiratory"]),
        "cardiovascular": _matching_columns(out.columns, ["cardiac", "cardiovascular", "heart", "stroke"]),
        "dehydration": _matching_columns(out.columns, ["diarrhea_with_dehydration", "dehydration"]),
        "heat_sensitive_illness": _matching_columns(out.columns, ["diarrhea", "gastro", "fever"]),
        "severe_outcomes": _matching_columns(out.columns, ["deaths_occurring_at_emergency", "deaths_occurring_at_sncu", "deaths_"], require_total=True),
    }

    # De-duplicate columns that match multiple groups.
    used = set()
    for k in groups:
        groups[k] = [c for c in groups[k] if not (c in used or used.add(c))]

    weights = {
        "respiratory": 0.30,
        "cardiovascular": 0.25,
        "dehydration": 0.20,
        "heat_sensitive_illness": 0.15,
        "severe_outcomes": 0.10,
    }

    component_scores = {}
    for group, group_cols in groups.items():
        if not group_cols:
            continue
        vals = []
        for c in group_cols:
            vals.append(_district_percentile(out[c]))
        component_scores[group] = pd.concat(vals, axis=1).mean(axis=1, skipna=True)

    if not component_scores:
        # Do not fabricate a district ranking when the HMIS file has no usable
        # heat-sensitive indicators.
        result = out[["district"]].drop_duplicates("district").copy()
        result["health_vulnerability_score"] = 50.0
        result["health_vulnerability_level"] = "UNKNOWN"
        result["vulnerability_data_status"] = "NO_HEAT_SENSITIVE_HMIS_INDICATORS"
        result["vulnerability_components"] = "none"
        return result[cols].drop_duplicates("district")

    available_weight = sum(weights[g] for g in component_scores)
    score01 = sum(weights[g] * component_scores[g] for g in component_scores) / available_weight

    result = out[["district"]].drop_duplicates("district").copy()
    # Keep one decimal but do not collapse the underlying score into a level.
    result["health_vulnerability_score"] = (score01 * 100).round(1)
    result["health_vulnerability_level"] = pd.cut(
        result["health_vulnerability_score"],
        [-np.inf, 20, 40, 60, 80, np.inf],
        labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"],
        include_lowest=True,
    ).astype("string")
    result["vulnerability_data_status"] = "HMIS_RELATIVE_BASELINE_PROXY"
    result["vulnerability_components"] = ", ".join(component_scores.keys())
    return result[cols].drop_duplicates("district")


def add_health_risk_features(df, baseline=None):
    """Attach health vulnerability without overwriting a real monthly HMIS score.

    If the caller already supplied health_vulnerability_score keyed by
    district-month from the monthly HMIS engine, preserve it. Otherwise fall
    back to the legacy static district baseline.
    """
    out = df.copy()

    # A monthly HMIS score is authoritative when already present.
    existing_score = None
    if "health_vulnerability_score" in out.columns:
        existing_score = pd.to_numeric(out["health_vulnerability_score"], errors="coerce")
        if existing_score.notna().any():
            out["health_vulnerability_score"] = existing_score.clip(0, 100).round(1)
            out["health_vulnerability_level"] = pd.cut(
                out["health_vulnerability_score"],
                [-np.inf, 20, 40, 60, 80, np.inf],
                labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"],
                include_lowest=True,
            ).astype("string").fillna("UNKNOWN")
            if "vulnerability_data_status" not in out.columns:
                out["vulnerability_data_status"] = "HMIS_MONTHLY_OR_EXISTING"
            if "vulnerability_components" not in out.columns:
                out["vulnerability_components"] = "monthly HMIS health indicators"
            return out

    # No real monthly score exists; use the legacy district baseline.
    for c in ["health_vulnerability_score", "health_vulnerability_level",
              "vulnerability_data_status", "vulnerability_components"]:
        if c in out.columns:
            out = out.drop(columns=[c])

    v = build_district_vulnerability(baseline)
    if "district" in out.columns and not v.empty:
        out = out.merge(v, on="district", how="left", validate="many_to_one")

    out["health_vulnerability_score"] = pd.to_numeric(
        out.get("health_vulnerability_score", pd.Series(50.0, index=out.index)),
        errors="coerce",
    ).fillna(50.0).clip(0, 100).round(1)

    out["health_vulnerability_level"] = pd.cut(
        out["health_vulnerability_score"],
        [-np.inf, 20, 40, 60, 80, np.inf],
        labels=["LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"],
        include_lowest=True,
    ).astype("string").fillna("UNKNOWN")
    out["vulnerability_data_status"] = out.get(
        "vulnerability_data_status", pd.Series("HMIS_RELATIVE_BASELINE_PROXY", index=out.index)
    ).fillna("HMIS_RELATIVE_BASELINE_PROXY")
    out["vulnerability_components"] = out.get(
        "vulnerability_components", pd.Series("heat-sensitive HMIS indicators", index=out.index)
    ).fillna("heat-sensitive HMIS indicators")
    return out
