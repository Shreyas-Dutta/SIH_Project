# Added final SIH features

1. Human Thermal Stress Index (HTSI) is already a named final metric in `features/thermal.py`.
2. GIS payload module added for district/ward dashboard integration. Ward support requires ward boundary + ward-level weather/demographic data.
3. True demographic vulnerability loader added. Use official Census/NFHS/state datasets and `data_templates/demographics_template.csv`.
4. Five-day forecast risk builder added.
5. FastAPI alert API added: `/health`, `/forecast`, `/alert/{district}`.
6. Actual mortality/hospitalization loader added. The project does NOT fabricate labels; dated real outcomes can be merged and used for supervised training.

Run API: `uvicorn heatwave_pipeline.api.app:app --reload`
