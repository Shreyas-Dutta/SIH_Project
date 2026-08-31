FINAL HEAT-HEALTH PIPELINE UPGRADE

This version fixes the 5-day forecast risk output failure caused by:
'Can only use .dt accessor with datetimelike values'

Changes:
- All weather/forecast dates are converted safely with pd.to_datetime.
- Invalid forecast dates are removed with a clear validation error if necessary.
- district_5day_risk_forecast.csv is generated.
- top_risk_districts.csv is generated.
- dashboard_summary.json is generated.

Run from the project folder:
python -m heatwave_pipeline.main

Expected:
[OK] Created 5-day district risk dashboard outputs.

The system does not create synthetic mortality labels. Health-impact scores remain
operational risk estimates based on available weather and HMIS-derived vulnerability data.

LIVE PREDICTION UPGRADE (2026-08-31)

The dashboard no longer treats district_5day_risk_forecast.csv as the primary
source of prediction data. It fetches a fresh 5-day Open-Meteo forecast and
recalculates the heat-health risk from that forecast.

Refresh behavior:
- Dashboard automatically refreshes its live prediction cache every 5 minutes.
- "Refresh latest prediction" forces an immediate new weather API fetch.
- Running `python -m heatwave_pipeline.main` forces a fresh 5-day forecast instead
  of accepting an old same-day forecast cache.
- `forecast_5_day_assam.csv` now records `fetched_at_ist` so the data freshness
  can be displayed/audited.

Important:
- The health-impact score is an operational heat-health risk estimate.
- It is not a mortality prediction unless dated outcome labels are trained into
  a separate validated supervised model.

ACCURACY / RISK-SCORE UPGRADE (v7)
----------------------------------
1. HTSI is now a normalized thermal-burden screening index using robust percentile-based components (WBGT, heat index, Tmax, hot nights, WBGT anomaly, persistence) instead of arbitrary fixed normalization.
2. Heatwave chance is now a model-based next-3-day likelihood (%) from a forecast-compatible logistic model trained on historical heatwave labels. The model uses only features available at forecast time.
3. Historical monthly/district climatology is used for Tmax/WBGT p90/p95 thresholds, with a monthly Assam fallback where district-specific history is unavailable. This avoids deriving heatwave thresholds from the 5-day forecast itself.
4. HIRL now correctly normalizes HTSI on a 0-100 scale and combines thermal burden, WBGT, persistence, hot nights, health vulnerability and heatwave likelihood.
5. HIRL remains an operational screening score, not a mortality probability. True health-outcome accuracy requires dated hospital/admission/mortality labels matched to weather exposure.
6. Current validation for the compact forecast model is stored in models/heatwave_forecast_compact_meta.json; the existing full historical model remains available for research/diagnostics.
