# Assam Heatwave Health-Impact Pipeline

## Run

From the project root:

```bash
python main.py
```

## Code structure

- `heatwave_pipeline/data/` — HMIS and weather ingestion
- `heatwave_pipeline/features/` — thermal calculations and feature engineering
- `heatwave_pipeline/dataset/` — merge, cleaning, imputation and ML dataset preparation
- `heatwave_pipeline/models/` — training and evaluation
- `heatwave_pipeline/utils/` — logging and shared helpers
- `data/` — recommended location for raw/processed datasets
- `output/` — recommended location for generated datasets, models and reports

### Important fix

The training failure:

`unsupported operand type(s) for |: 'set' and 'list'`

was fixed by converting `TARGET_COLUMNS` to a set before combining it with other sets.

The existing CSV files are left in place for compatibility with your current `config.py` and cache logic.


## Upgraded outputs
- `models/confusion_matrix_heatwave_next_3d.csv`
- `models/feature_importance_heatwave_next_3d.csv`
- `district_5day_risk_forecast.csv`
- `top_risk_districts.csv`
- `dashboard_summary.json`

Run from the project root:
```bash
python -m heatwave_pipeline.main
```

## Live/latest prediction behavior

The dashboard is now configured to use the **latest available Open-Meteo 5-day
forecast** as its primary prediction input. It does not rely on a previously
generated `district_5day_risk_forecast.csv` for the displayed prediction.

- Automatic refresh window: **5 minutes**.
- Use **Refresh latest prediction** for an immediate refresh.
- Running `python -m heatwave_pipeline.main` forces a fresh forecast API fetch.
- `forecast_5_day_assam.csv` stores `fetched_at_ist` for freshness/auditability.

This means the prediction changes when the upstream weather forecast changes,
without requiring you to manually regenerate the historical ML dataset.
