from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from heatwave_pipeline.data.hmis_monthly import load_monthly_hmis, build_monthly_vulnerability

print("=" * 72)
print("HMIS MONTHLY HEALTH VULNERABILITY TEST")
print("=" * 72)
raw = load_monthly_hmis()
if raw is None:
    print("STATUS: NO MONTHLY HMIS FILES FOUND")
    print("Expected location: hmis_monthly/*.csv")
    print("The pipeline will NOT invent monthly vulnerability from the static HMIS snapshot.")
    raise SystemExit(0)

monthly = build_monthly_vulnerability(raw)
if monthly is None or monthly.empty:
    print("STATUS: MONTHLY FILES FOUND, BUT COULD NOT BUILD VULNERABILITY")
    raise SystemExit(1)

print(f"Monthly vulnerability rows: {len(monthly):,}")
print(f"Districts: {monthly['district'].nunique()}")
print(f"Months: {monthly['report_month'].nunique()}")
print(f"Score min: {monthly['health_vulnerability_score'].min():.1f}")
print(f"Score max: {monthly['health_vulnerability_score'].max():.1f}")
print(f"Score unique: {monthly['health_vulnerability_score'].nunique()}")
print(f"District-month unique keys: {monthly[['district','report_month']].drop_duplicates().shape[0]:,}")

if monthly['health_vulnerability_score'].nunique() <= 1:
    print("FAIL: score is still constant")
    raise SystemExit(2)

monthly.to_csv(ROOT / 'hmis_monthly_vulnerability.csv', index=False, encoding='utf-8-sig')
print("PASS: monthly HMIS vulnerability is time-varying")
print(monthly.head(20).to_string(index=False))
