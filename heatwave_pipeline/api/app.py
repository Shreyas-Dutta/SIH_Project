from pathlib import Path
import pandas as pd
from fastapi import FastAPI

app=FastAPI(title="Heat-Health Alert API")
BASE=Path(__file__).resolve().parents[2]
OUTPUT=BASE/"district_5day_risk_forecast.csv"
SUMMARY=BASE/"dashboard_summary.json"

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Assam Heat-Health Risk API is running"
    }

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/forecast")
def forecast():
    if not OUTPUT.exists(): return {"error":"Run forecast pipeline first"}
    return pd.read_csv(OUTPUT).to_dict(orient="records")

@app.get("/alert/{district}")
def alert(district:str):
    if not OUTPUT.exists(): return {"error":"Forecast unavailable"}
    df=pd.read_csv(OUTPUT); x=df[df.district.astype(str).str.lower()==district.lower()]
    if x.empty:return {"error":"District not found"}
    return {"district":district,"alerts":x.to_dict(orient="records")}

@app.get("/summary")
def summary():
    if not SUMMARY.exists():
        return {"error":"Run pipeline first"}
    import json
    return json.loads(SUMMARY.read_text(encoding="utf-8"))

