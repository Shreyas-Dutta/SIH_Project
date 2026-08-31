# Deployment

## Streamlit Community Cloud
- Repository root: project root
- Main file: `dashboard/app.py`
- Python version: 3.11

## Railway / Render
Start command:
`streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port $PORT`

## API (optional separate service)
Start command:
`uvicorn heatwave_pipeline.api.app:app --host 0.0.0.0 --port $PORT`

The dashboard expects the included trained models and CSV forecast/data files to remain in the repository.
