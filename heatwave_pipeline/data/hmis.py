import re
import numpy as np
import pandas as pd
from ..config import BASE_DIR, HMIS_FILE, HMIS_RESOURCE_ID, HMIS_RESOURCE_PAGE, DISTRICTS
from ..utils.helpers import clean_text, slugify, safe_numeric, save_csv, log

DISTRICT_ALIASES = {
    "kamrup metropolitan": "Kamrup M",
    "kamrup metro": "Kamrup M",
    "kamrup m": "Kamrup M",
    "kamrup rural": "Kamrup R",
    "kamrup r": "Kamrup R",
    "marigaon": "Marigaon",
    "morigaon": "Marigaon",
    "sibsagar": "Sibsagar",
    "sivasagar": "Sibsagar",
    "dima hasao": "Dima Hasao",
    "dimahasao": "Dima Hasao",
    "north cachar hills": "Dima Hasao",
}

def find_hmis_header(path):
    """
    The downloaded HMIS CSV can contain title/metadata rows before
    the actual table header. Find the row containing:
        Indicator / S.No. / Parameters / Type / District - ...
    """
    raw = pd.read_csv(
        path,
        header=None,
        dtype=str,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip",
    )

    max_scan = min(len(raw), 80)

    best_row = None
    best_score = -1

    for i in range(max_scan):
        values = [clean_text(v).lower() for v in raw.iloc[i].tolist()]
        joined = " | ".join(values)

        score = 0
        if "indicator" in joined:
            score += 3
        if "parameters" in joined:
            score += 3
        if "s.no" in joined or "sno" in joined:
            score += 2
        if "district -" in joined:
            score += 5
        if "assam" in joined:
            score += 2

        if score > best_score:
            best_score = score
            best_row = i

    if best_row is None or best_score < 5:
        # Fall back to pandas normal header.
        return None

    return best_row

def load_hmis():
    log("")
    log("=" * 70)
    log("1. LOADING HMIS HEALTH DATA")
    log("=" * 70)

    if not HMIS_FILE.exists():
        log("[ERROR] hmis_data.csv was not found.")
        log("")
        log("Put the manually downloaded HMIS CSV here:")
        log(str(HMIS_FILE))
        log("")
        log("HMIS resource:")
        log(HMIS_RESOURCE_PAGE)
        log(f"HMIS resource ID: {HMIS_RESOURCE_ID}")
        return None

    log(f"[INFO] Reading: {HMIS_FILE}")

    header_row = find_hmis_header(HMIS_FILE)

    if header_row is None:
        log("[WARNING] Could not confidently detect a metadata header.")
        header_row = 0
    else:
        log(f"[OK] HMIS table header detected at CSV row {header_row + 1}")

    df = pd.read_csv(
        HMIS_FILE,
        header=header_row,
        dtype=str,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip",
    )

    # Remove completely empty columns/rows.
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Clean duplicate/unnamed headers.
    new_columns = []
    seen = {}

    for col in df.columns:
        col = clean_text(col)
        if not col:
            col = "unnamed"

        count = seen.get(col, 0)
        seen[col] = count + 1

        if count:
            col = f"{col}_{count + 1}"

        new_columns.append(col)

    df.columns = new_columns

    # Remove rows that accidentally repeat the header.
    first_col_text = df.iloc[:, 0].astype(str).str.lower()

    repeated_header = first_col_text.str.contains(
        r"^indicator$", regex=True, na=False
    )
    df = df.loc[~repeated_header].copy()

    log(f"[OK] HMIS rows loaded: {len(df):,}")
    log(f"[OK] HMIS columns loaded: {len(df.columns):,}")

    save_csv(df, "hmis_health_raw.csv")

    return df

def canonical_district(name):
    name = clean_text(name)

    if not name:
        return None

    low = re.sub(r"\s+", " ", name.lower()).strip()

    if low in DISTRICT_ALIASES:
        return DISTRICT_ALIASES[low]

    for district in DISTRICTS:
        if low == district.lower():
            return district

    return None

def parse_district_column(column):
    """
    Examples:
        District - Kamrup M - Total [(A+B) or (C+D)]
        District - Kamrup M - Public [A]
        District - Kamrup M - Private [B]
        District - Kamrup M - Urban [C]
        District - Kamrup M - Rural [D]
    """
    text = clean_text(column)

    if not text.lower().startswith("district"):
        return None, None

    # Remove the prefix.
    rest = re.sub(r"^\s*district\s*-\s*", "", text, flags=re.I)

    # Match against known district names, longest first.
    for district in sorted(DISTRICTS, key=len, reverse=True):
        pattern = r"^" + re.escape(district) + r"\s*-\s*(.*)$"
        match = re.match(pattern, rest, flags=re.I)

        if match:
            category = clean_text(match.group(1))

            category_low = category.lower()

            if "total" in category_low:
                category = "Total"
            elif "public" in category_low:
                category = "Public"
            elif "private" in category_low:
                category = "Private"
            elif "urban" in category_low:
                category = "Urban"
            elif "rural" in category_low:
                category = "Rural"
            else:
                category = slugify(category)

            return district, category

    # Generic fallback for district columns not in the list.
    match = re.match(
        r"^(.+?)\s*-\s*(total|public|private|urban|rural)",
        rest,
        flags=re.I,
    )

    if match:
        district = canonical_district(match.group(1))
        if district:
            return district, match.group(2).title()

    return None, None

def identify_hmis_columns(df):
    district_columns = []

    for col in df.columns:
        district, category = parse_district_column(col)

        if district and category:
            district_columns.append(
                {
                    "column": col,
                    "district": district,
                    "category": category,
                }
            )

    return district_columns

def find_column(df, candidates):
    lower_map = {clean_text(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    # Partial match.
    for col in df.columns:
        low = clean_text(col).lower()
        for candidate in candidates:
            if candidate.lower() in low:
                return col

    return None

def extract_hmis_indicators(df):
    """
    Converts the wide HMIS structure into:
        indicator
        parameter
        type
        district
        category
        value

    Only rows that look health-relevant are retained for the ML
    baseline. The complete raw HMIS data remains available separately.
    """
    log("")
    log("=" * 70)
    log("2. EXTRACTING HMIS INDICATORS")
    log("=" * 70)

    district_columns = identify_hmis_columns(df)

    if not district_columns:
        log("[ERROR] No district columns found in HMIS.")
        log("[INFO] First 20 columns:")
        for c in list(df.columns)[:20]:
            log(f"  {c}")
        return None

    log(f"[OK] District-value columns found: {len(district_columns):,}")

    indicator_col = find_column(
        df,
        ["Indicator", "indicator"],
    )

    parameter_col = find_column(
        df,
        ["Parameters", "Parameter", "parameters"],
    )

    type_col = find_column(
        df,
        ["Type", "type"],
    )

    sno_col = find_column(
        df,
        ["S.No.", "S.No", "Sno", "S No"],
    )

    if indicator_col is None:
        # First non-district metadata column.
        district_names = {x["column"] for x in district_columns}
        metadata = [c for c in df.columns if c not in district_names]
        indicator_col = metadata[0] if metadata else df.columns[0]

    if parameter_col is None:
        parameter_col = indicator_col

    records = []

    for row_index, row in df.iterrows():
        indicator = clean_text(row.get(indicator_col, ""))
        parameter = clean_text(row.get(parameter_col, ""))
        type_value = clean_text(row.get(type_col, "")) if type_col else ""
        sno = clean_text(row.get(sno_col, "")) if sno_col else ""

        # Ignore completely blank metadata rows.
        if not indicator and not parameter:
            continue

        row_text = " ".join(
            [indicator, parameter, type_value]
        ).lower()

        # Broad health-impact keywords. This is intentionally broad;
        # we save the full normalized indicator table too.
        health_keywords = [
            "death",
            "deaths",
            "mortality",
            "admission",
            "admissions",
            "admitted",
            "inpatient",
            "ipd",
            "outpatient",
            "opd",
            "hospital",
            "patient",
            "disease",
            "illness",
            "fever",
            "respiratory",
            "stroke",
            "cardiac",
            "cardiovascular",
            "emergency",
            "icu",
            "casualty",
            "morbidity",
        ]

        relevant = any(k in row_text for k in health_keywords)

        for item in district_columns:
            raw_value = row.get(item["column"], np.nan)
            numeric_value = safe_numeric(
                pd.Series([raw_value])
            ).iloc[0]

            # Keep only numeric observations.
            if pd.isna(numeric_value):
                continue

            records.append(
                {
                    "source_row": row_index,
                    "s_no": sno,
                    "indicator": indicator,
                    "parameter": parameter,
                    "type": type_value,
                    "district": item["district"],
                    "category": item["category"],
                    "value": numeric_value,
                    "health_relevant": int(relevant),
                }
            )

    long_df = pd.DataFrame(records)

    if long_df.empty:
        log("[ERROR] No numeric HMIS district observations were extracted.")
        return None

    # Keep all extracted values in a normalized file.
    save_csv(long_df, "hmis_health_indicators.csv")

    relevant_df = long_df[long_df["health_relevant"] == 1].copy()

    if relevant_df.empty:
        log(
            "[WARNING] No obvious death/admission/health-impact "
            "keywords were found."
        )
        relevant_df = long_df.copy()

    log(
        f"[OK] Normalized HMIS observations: "
        f"{len(long_df):,}"
    )
    log(
        f"[OK] Health-relevant observations: "
        f"{len(relevant_df):,}"
    )
    log(
        f"[OK] Districts represented: "
        f"{relevant_df['district'].nunique()}"
    )

    return relevant_df

def build_hmis_baseline(indicators):
    """
    HMIS is aggregate data and generally has no daily date in this
    resource. Therefore we construct one row per district containing
    baseline health indicators.

    These are baseline features, NOT daily mortality labels.
    """
    log("")
    log("=" * 70)
    log("3. BUILDING DISTRICT HEALTH BASELINE")
    log("=" * 70)

    if indicators is None or indicators.empty:
        return pd.DataFrame({"district": list(DISTRICTS.keys())})

    work = indicators.copy()

    # Limit the number of indicator columns so that a very wide HMIS
    # file does not create thousands of unusable ML columns.
    # Prefer mortality/admission/hospital indicators.
    work["priority"] = 0

    text = (
        work["indicator"].fillna("").astype(str)
        + " "
        + work["parameter"].fillna("").astype(str)
    ).str.lower()

    priority_terms = {
        "death": 10,
        "mortality": 10,
        "admission": 9,
        "admitted": 9,
        "hospital": 8,
        "inpatient": 8,
        "ipd": 8,
        "patient": 5,
        "disease": 4,
        "respiratory": 4,
        "cardiac": 4,
        "stroke": 4,
        "fever": 3,
    }

    for term, score in priority_terms.items():
        work.loc[text.str.contains(term, regex=False), "priority"] += score

    # Keep rows with some health relevance.
    work = work[work["priority"] > 0].copy()

    if work.empty:
        log("[WARNING] No prioritized HMIS indicators available.")
        return pd.DataFrame({"district": list(DISTRICTS.keys())})

    # Rank indicators globally.
    indicator_scores = (
        work.groupby(["indicator", "parameter"], dropna=False)["priority"]
        .max()
        .reset_index()
        .sort_values(
            ["priority", "indicator", "parameter"],
            ascending=[False, True, True],
        )
    )

    # Up to 40 useful indicator/parameter combinations.
    selected = indicator_scores.head(40)

    selected_keys = set(
        zip(selected["indicator"], selected["parameter"])
    )

    work = work[
        work.apply(
            lambda r: (r["indicator"], r["parameter"]) in selected_keys,
            axis=1,
        )
    ].copy()

    # Aggregate duplicate rows by district/category/indicator.
    pivot = (
        work.groupby(
            ["district", "indicator", "parameter", "category"],
            dropna=False,
        )["value"]
        .mean()
        .reset_index()
    )

    pivot["feature_base"] = (
        pivot["indicator"].fillna("").astype(str)
        + "_"
        + pivot["parameter"].fillna("").astype(str)
        + "_"
        + pivot["category"].fillna("").astype(str)
    )

    pivot["feature"] = pivot["feature_base"].map(slugify)

    # Pivot into district-level baseline.
    baseline = pivot.pivot_table(
        index="district",
        columns="feature",
        values="value",
        aggfunc="mean",
    ).reset_index()

    baseline.columns.name = None

    # Prefix all HMIS fields.
    baseline = baseline.rename(
        columns={
            c: (
                c if c == "district"
                else f"hmis_{c}"
            )
            for c in baseline.columns
        }
    )

    # Make sure every Assam HMIS district exists.
    all_districts = pd.DataFrame(
        {"district": list(DISTRICTS.keys())}
    )

    baseline = all_districts.merge(
        baseline,
        on="district",
        how="left",
    )

    save_csv(baseline, "hmis_district_baseline.csv")

    log(
        f"[OK] District baseline rows: {len(baseline):,}"
    )
    log(
        f"[OK] HMIS baseline features: "
        f"{len(baseline.columns) - 1:,}"
    )

    return baseline
