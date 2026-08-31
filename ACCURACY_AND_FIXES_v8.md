# Accuracy + reliability upgrade v8

## Fixed runtime error
`PermissionError: [Errno 13]` on `heatwave_ml_dataset.csv` is a Windows file-lock issue, commonly caused by Excel/OneDrive/another process holding the CSV open. `save_csv()` is now resilient: it writes atomically and, if the destination is locked, saves a timestamped fresh copy and continues instead of aborting the pipeline.

## Heatwave probability
- Live forecasting now uses the model trained during the current pipeline run instead of silently using an older static model artifact.
- The live artifact is also refreshed as `models/heatwave_forecast_compact.joblib` with matching feature metadata.
- Forecast probability is based only on weather/climatology features available at forecast time.
- The forecast event evidence is forward-looking (next 1–3 days), not a past rolling window.
- Probability is a model likelihood/screening estimate, not a medically calibrated probability.

## HTSI
HTSI is a 0–100 screening score based on district-month thermal extremes rather than whole-year percentile ranks. It combines:
- WBGT burden: 35%
- Heat-index burden: 18%
- Tmax burden: 17%
- Hot-night burden: 10%
- WBGT anomaly: 10%
- Persistence: 10%

This reflects that heat-health burden depends on intensity, duration and population vulnerability.

## TRL
TRL is derived directly from HTSI:
- 0–20 Low
- 20–40 Moderate
- 40–60 High
- 60–80 Very High
- 80–100 Extreme

## HIRL
HIRL is an operational screening score, not a mortality probability. It combines thermal exposure, WBGT, persistence, hot nights, vulnerability and heatwave likelihood. Vulnerability modifies the exposure signal rather than replacing it.

## Model validity improvement
The heatwave model is now restricted to weather/climatology features that can actually be supplied by the live forecast. HMIS outcome variables and unavailable future/lags are not used as predictive inputs. Logistic regression is allowed to output natural probabilities (`class_weight=None`) while the alert threshold is selected separately on the validation period.

The v8 training run produced a test ROC-AUC around 0.91 and test PR-AUC around 0.41 for the weather-only 3-day heatwave target. These are retrospective metrics on the included historical weather dataset, not guarantees of future performance.
