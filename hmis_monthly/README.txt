MONTHLY HMIS INPUT
==================
Place the health-indicator-wise monthly HMIS CSV files here.

Required logical fields:
- district
- sub_district / facility (if available)
- indicator / parameter / HMIS key
- value
- reporting month (or year + month)

The pipeline aggregates sub-district/facility observations to district-month,
calculates a 0-100 relative health vulnerability score, derives the level,
and joins it to daily weather using district + year-month.

Do NOT replace this with the static hmis_data.csv. The static file has no
month key and is used only as a fallback baseline.
