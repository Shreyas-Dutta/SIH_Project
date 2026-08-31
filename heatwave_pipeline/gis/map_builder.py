import pandas as pd

def export_gis_payload(risk_df, output_path):
    cols=[c for c in ["district","date","health_impact_risk_score","health_impact_risk_level","human_thermal_stress_index","thermal_risk_level"] if c in risk_df]
    risk_df[cols].to_json(output_path,orient="records",date_format="iso")
