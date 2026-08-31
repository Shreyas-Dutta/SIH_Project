"""Transparent impact-risk scorer.

This is an operational risk score, NOT a mortality prediction model unless a
separate supervised model is trained on dated mortality/admission outcomes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .alerts import build_advisory


def _norm(s, lo, hi):
    return ((pd.to_numeric(s, errors="coerce") - lo) / (hi - lo)).clip(0, 1).fillna(0)


def score_health_impact(df: pd.DataFrame, heat_probability_col: str | None = None) -> pd.DataFrame:
    """Operational heat-health screening score (0-100), not mortality probability."""
    out = df.copy()
    htsi = _norm(out.get("human_thermal_stress_index", pd.Series(0, index=out.index)), 0, 100)
    wbgt = _norm(out.get("wbgt_estimated_c", pd.Series(0, index=out.index)), 24, 34)
    hot_streak = _norm(out.get("hot_day_streak", pd.Series(0, index=out.index)), 0, 3)
    hot_night = _norm(out.get("hot_night_3d_count", pd.Series(0, index=out.index)), 0, 3)
    vulnerability = _norm(out.get("health_vulnerability_score", pd.Series(50, index=out.index)), 0, 100)
    heat_prob = (pd.to_numeric(out[heat_probability_col], errors="coerce").clip(0, 100) / 100.0).fillna(0) if heat_probability_col and heat_probability_col in out.columns else pd.Series(0.0, index=out.index)

    # WHO describes heat-health impact as a function of intensity, duration,
    # acclimatization/adaptation and population vulnerability. The score therefore
    # gives the largest weight to current thermal exposure, then persistence and vulnerability.
    exposure = 0.65 * htsi + 0.20 * wbgt + 0.10 * hot_streak + 0.05 * hot_night
    modifier = 0.70 + 0.20 * vulnerability + 0.10 * heat_prob
    score = (100 * exposure * modifier).clip(0, 100)

    out["health_impact_risk_score"] = score.round(1)
    out["health_impact_risk_level"] = pd.cut(
        out["health_impact_risk_score"], [-np.inf, 25, 50, 75, np.inf],
        labels=["GREEN", "YELLOW", "ORANGE", "RED"]
    ).astype(str)
    out["advisory_actions"] = out["health_impact_risk_level"].map(lambda x: " | ".join(build_advisory(x)))
    out["risk_note"] = "Operational heat-health screening score; not a mortality probability."
    return out
