import pandas as pd

REQUIRED={"district","date"}

def load_health_outcomes(path):
    df=pd.read_csv(path); df.columns=[c.strip().lower() for c in df.columns]
    missing=REQUIRED-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {missing}")
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    return df.dropna(subset=["date"])

def merge_health_outcomes(weather_df,outcomes_df):
    return weather_df.merge(outcomes_df,on=["district","date"],how="left")
