# SIH Heat-Health Risk Architecture v9

## End-to-end architecture

1. **Data ingestion**
   - Historical weather: daily district weather observations.
   - Live forecast: latest 5-day forecast from the weather API.
   - HMIS: district health/vulnerability baseline.

2. **Data quality and harmonisation**
   - Standardise district names and dates.
   - Remove duplicate district/date records.
   - Convert numeric fields and handle missing values.
   - Historical training data and live forecast data use the same canonical weather schema.

3. **Thermal feature engine**
   - Heat Index, wet-bulb estimate and an explicitly labelled estimated WBGT proxy.
   - District-month climatology: p90/p95 thresholds and anomalies.
   - Persistence features and HTSI (0-100).

4. **Heatwave target construction**
   - A future heatwave day is a local extreme in Tmax OR estimated WBGT.
   - `heatwave_next_3d = 1` when at least 2 of the next 3 days meet the extreme criterion.
   - The target is weather-derived; HMIS outcomes are not used as future predictors.

5. **Leakage-safe ML training**
   - Chronological 70/15/15 train/validation/test split.
   - Only the training partition is balanced using deterministic year-stratified negative undersampling at 3:1 negative:positive.
   - Validation and test retain the real class prevalence.
   - Probability prior correction restores the original event prior after undersampling.
   - Model selection uses validation PR-AUC, with F1 fallback.
   - Warning threshold is selected on validation data, never test data.

6. **Live prediction**
   - Fetch latest forecast.
   - Recompute thermal/climatology features.
   - Apply the saved best model.
   - Produce heatwave probability and forecast-event evidence.

7. **Health-impact layer**
   - Combine HTSI, WBGT, persistence, hot nights, health vulnerability and heatwave probability into HIRL.
   - HIRL is an operational screening score, not a mortality probability.

8. **Outputs**
   - 5-day district risk CSV.
   - GIS payload.
   - Alerts and advisories.
   - Streamlit dashboard.

## Why this architecture is better

- It fixes severe class imbalance without leaking synthetic samples across time.
- It keeps validation/test prevalence realistic, so reported metrics are meaningful.
- It prevents a high accuracy score from hiding poor minority-class recall.
- It separates **weather-event prediction** from **health-impact scoring**.
- It keeps the live system forecast-compatible: the model only uses features available at prediction time.
