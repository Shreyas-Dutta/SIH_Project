# Heat-Health GIS Dashboard
Run the pipeline first so forecast risk outputs exist:

```bash
python -m heatwave_pipeline.main
```

Then start the dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard reads the generated 5-day district forecast, shows risk metrics, an interactive GIS marker map when latitude/longitude are available, a risk table, top-risk districts, and alert previews.
