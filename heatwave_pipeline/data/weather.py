import time
import pandas as pd
import requests

from ..config import (
    BASE_DIR,
    START_DATE,
    END_DATE,
    DISTRICTS,
    DISTRICT_COORDS,
)
from ..utils.helpers import log


# ============================================================
# CONFIGURATION
# ============================================================

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_FILE = BASE_DIR / "historical_weather_assam.csv"
FORECAST_FILE = BASE_DIR / "forecast_5_day_assam.csv"

TIMEOUT = 90

# Do not blindly retry 429 responses.
# The server can rate-limit clients for a while.
MAX_RETRIES = 3

RETRY_DELAY = 10

# Small multi-location batches are safer than one request per district
# and safer than one enormous 27-location request.
DISTRICT_BATCH_SIZE = 5


# ============================================================
# REQUIRED DAILY VARIABLES
#
# Keep this list limited to variables actually used by the
# heatwave pipeline. Avoid unsupported daily variables.
# ============================================================

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",

    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",

    "dew_point_2m_mean",

    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",

    "precipitation_sum",
    "rain_sum",

    "wind_speed_10m_max",
    "wind_direction_10m_dominant",

    "shortwave_radiation_sum",

    "surface_pressure_mean",
]


# ============================================================
# NORMALIZE DISTRICT
# ============================================================

def normalize_district(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
    )


# ============================================================
# GET DISTRICT COORDINATES
# ============================================================

def get_coordinates(district):

    target = normalize_district(district)

    for name, coords in DISTRICT_COORDS.items():

        if normalize_district(name) == target:
            return coords

    return None


# ============================================================
# REMOVE DUPLICATE COLUMNS SAFELY
#
# Important:
# pandas returns a DataFrame rather than Series when duplicate
# column names exist. This was the cause of:
#
# TypeError: arg must be a list, tuple, 1-d array, or Series
# ============================================================

def clean_duplicate_columns(df):

    if df is None or df.empty:
        return df

    if not df.columns.duplicated().any():
        return df

    log(
        "[CACHE] Duplicate columns detected. "
        "Cleaning them safely..."
    )

    cleaned = pd.DataFrame(index=df.index)

    # Preserve original column order.
    for column in dict.fromkeys(df.columns):

        same = df.loc[
            :,
            df.columns == column
        ]

        if same.shape[1] == 1:

            cleaned[column] = (
                same.iloc[:, 0]
            )

        else:

            # Keep the first non-null value from duplicate columns.
            cleaned[column] = (
                same.bfill(axis=1).iloc[:, 0]
            )

    return cleaned


# ============================================================
# LOAD CACHE
# ============================================================

def load_weather_cache():

    if not WEATHER_FILE.exists():

        log(
            "[CACHE] historical_weather_assam.csv "
            "does not exist."
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            WEATHER_FILE,
            low_memory=False,
        )

    except Exception as exc:

        log(
            f"[WARNING] Could not read weather cache: "
            f"{exc}"
        )

        return pd.DataFrame()

    df = clean_duplicate_columns(df)

    if (
        "district" not in df.columns
        or "date" not in df.columns
    ):

        log(
            "[WARNING] Weather cache is missing "
            "district/date columns."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df["district"] = (
        df["district"]
        .astype(str)
        .str.strip()
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "district",
            "date",
        ]
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in DAILY_VARIABLES:

        if column not in df.columns:
            continue

        # Guaranteed Series after duplicate-column cleanup.
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Duplicate district/date rows
    # --------------------------------------------------------

    duplicate_count = (
        df.duplicated(
            subset=[
                "district",
                "date",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count:

        log(
            f"[CACHE] Removing "
            f"{duplicate_count:,} duplicate district/date rows."
        )

        df = df.drop_duplicates(
            subset=[
                "district",
                "date",
            ],
            keep="last",
        )

    df = df.sort_values(
        [
            "district",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    log(
        f"[CACHE] Loaded "
        f"{len(df):,} rows."
    )

    return df


# ============================================================
# CHECK ONE WEATHER ROW
# ============================================================

def row_is_complete(row):

    for column in DAILY_VARIABLES:

        if column not in row.index:
            return False

        if pd.isna(row[column]):
            return False

    return True


# ============================================================
# FIND MISSING DATES FOR ONE DISTRICT
# ============================================================

def get_missing_dates(
    cache,
    district,
):

    expected = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    if cache is None or cache.empty:
        return expected

    rows = cache[
        cache["district"].map(
            normalize_district
        )
        ==
        normalize_district(district)
    ].copy()

    if rows.empty:
        return expected

    valid_dates = set()

    for _, row in rows.iterrows():

        date = row["date"].normalize()

        if row_is_complete(row):

            valid_dates.add(
                date
            )

    return pd.DatetimeIndex([
        date
        for date in expected
        if date not in valid_dates
    ])


# ============================================================
# CHECK ENTIRE HISTORICAL CACHE
# ============================================================

def is_complete(cache):

    if cache is None or cache.empty:
        return False

    required = [
        "district",
        "date",
    ] + DAILY_VARIABLES

    for column in required:

        if column not in cache.columns:

            log(
                f"[CHECK] Missing cache column: "
                f"{column}"
            )

            return False

    for district in DISTRICTS:

        missing = get_missing_dates(
            cache,
            district,
        )

        if len(missing) > 0:
            return False

    return True


# ============================================================
# MAKE CONTIGUOUS DATE RANGES
# ============================================================

def contiguous_ranges(dates):

    if len(dates) == 0:
        return []

    dates = sorted(
        pd.Timestamp(x).normalize()
        for x in dates
    )

    ranges = []

    start = dates[0]
    previous = dates[0]

    for current in dates[1:]:

        if (
            current
            ==
            previous + pd.Timedelta(days=1)
        ):

            previous = current

        else:

            ranges.append(
                (
                    start,
                    previous,
                )
            )

            start = current
            previous = current

    ranges.append(
        (
            start,
            previous,
        )
    )

    return ranges


# ============================================================
# REQUEST OPEN-METEO
# ============================================================

def request_batch(
    districts,
    start_date,
    end_date,
):

    coordinates = []
    valid_districts = []

    for district in districts:

        coords = get_coordinates(
            district
        )

        if coords is None:

            log(
                f"[WARNING] Coordinates not found "
                f"for {district}"
            )

            continue

        coordinates.append(coords)
        valid_districts.append(district)

    if not valid_districts:
        return pd.DataFrame()

    params = {

        "latitude":
            ",".join(
                str(x[0])
                for x in coordinates
            ),

        "longitude":
            ",".join(
                str(x[1])
                for x in coordinates
            ),

        "start_date":
            start_date.strftime(
                "%Y-%m-%d"
            ),

        "end_date":
            end_date.strftime(
                "%Y-%m-%d"
            ),

        "daily":
            ",".join(
                DAILY_VARIABLES
            ),

        "timezone":
            "Asia/Kolkata",

        "temperature_unit":
            "celsius",

        "wind_speed_unit":
            "kmh",

        "precipitation_unit":
            "mm",

    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            log(
                f"[API] Requesting "
                f"{len(valid_districts)} districts: "
                f"{start_date.date()} -> "
                f"{end_date.date()}"
            )

            response = requests.get(
                ARCHIVE_URL,
                params=params,
                timeout=TIMEOUT,
            )

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                if retry_after:

                    try:
                        wait = int(
                            retry_after
                        )
                    except ValueError:
                        wait = 60

                else:

                    wait = 60 * attempt

                log(
                    f"[RATE LIMIT] HTTP 429. "
                    f"Waiting {wait} seconds."
                )

                time.sleep(
                    wait
                )

                continue

            # ------------------------------------------------
            # HTTP ERROR
            # ------------------------------------------------

            if response.status_code >= 400:

                try:

                    error_data = (
                        response.json()
                    )

                    reason = (
                        error_data.get(
                            "reason",
                            response.text[:300],
                        )
                    )

                except Exception:

                    reason = (
                        response.text[:300]
                    )

                log(
                    f"[API ERROR] HTTP "
                    f"{response.status_code}: "
                    f"{reason}"
                )

                # A 400 is normally a bad request and
                # should NOT be retried.
                if response.status_code == 400:

                    return pd.DataFrame()

                if attempt < MAX_RETRIES:

                    time.sleep(
                        RETRY_DELAY * attempt
                    )

                continue

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            payload = response.json()

            if isinstance(
                payload,
                dict,
            ):

                locations = [
                    payload
                ]

            elif isinstance(
                payload,
                list,
            ):

                locations = payload

            else:

                log(
                    "[ERROR] Unexpected API response."
                )

                return pd.DataFrame()

            frames = []

            for index, location in enumerate(
                locations
            ):

                if index >= len(
                    valid_districts
                ):
                    break

                district = (
                    valid_districts[index]
                )

                daily = location.get(
                    "daily"
                )

                if not daily:

                    log(
                        f"[WARNING] No daily data "
                        f"for {district}"
                    )

                    continue

                times = daily.get(
                    "time",
                    [],
                )

                frame = pd.DataFrame(
                    {
                        "date":
                            pd.to_datetime(
                                times,
                                errors="coerce",
                            ),

                        "district":
                            district,
                    }
                )

                for variable in DAILY_VARIABLES:

                    values = daily.get(
                        variable
                    )

                    if values is None:

                        frame[
                            variable
                        ] = pd.NA

                    else:

                        frame[
                            variable
                        ] = values

                # ------------------------------------------------
                # Sanity-check returned data.
                # ------------------------------------------------

                if not frame.empty:

                    frame = frame.dropna(
                        subset=[
                            "date"
                        ]
                    )

                    frames.append(
                        frame
                    )

            if not frames:

                return pd.DataFrame()

            result = pd.concat(
                frames,
                ignore_index=True,
            )

            return result

        except requests.exceptions.Timeout:

            log(
                f"[WARNING] Request timeout "
                f"(attempt {attempt}/"
                f"{MAX_RETRIES})."
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY * attempt
                )

        except requests.exceptions.RequestException as exc:

            log(
                f"[WARNING] Request error "
                f"(attempt {attempt}/"
                f"{MAX_RETRIES}): {exc}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY * attempt
                )

        except Exception as exc:

            log(
                f"[ERROR] Unexpected API error: "
                f"{exc}"
            )

            return pd.DataFrame()

    return pd.DataFrame()


# ============================================================
# MERGE NEW DATA INTO CACHE
# ============================================================

def merge_into_cache(
    cache,
    new_data,
):

    if new_data is None or new_data.empty:

        return cache

    if cache is None or cache.empty:

        result = new_data.copy()

    else:

        result = pd.concat(
            [
                cache,
                new_data,
            ],
            ignore_index=True,
        )

    # Remove duplicate columns if API/cache ever introduces them.
    result = clean_duplicate_columns(
        result
    )

    result["district"] = (
        result["district"]
        .astype(str)
        .str.strip()
    )

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    for column in DAILY_VARIABLES:

        if column not in result.columns:
            continue

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=[
            "district",
            "date",
        ]
    )

    result = result.drop_duplicates(
        subset=[
            "district",
            "date",
        ],
        keep="last",
    )

    return result.sort_values(
        [
            "district",
            "date",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# SAVE CACHE
# ============================================================

def save_cache(cache):

    cache = clean_duplicate_columns(
        cache
    )

    # Safe numeric conversion.
    for column in DAILY_VARIABLES:

        if column not in cache.columns:
            continue

        cache[column] = pd.to_numeric(
            cache[column],
            errors="coerce",
        )

    cache.to_csv(
        WEATHER_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    log(
        f"[CACHE] Saved "
        f"{len(cache):,} rows -> "
        f"{WEATHER_FILE.name}"
    )


# ============================================================
# FETCH ONLY MISSING HISTORICAL DATA
# ============================================================

def fetch_weather():

    log("")
    log("=" * 70)
    log("HISTORICAL WEATHER")
    log("=" * 70)

    # --------------------------------------------------------
    # LOAD CACHE FIRST
    # --------------------------------------------------------

    cache = load_weather_cache()

    # --------------------------------------------------------
    # COMPLETE = ZERO REQUESTS
    # --------------------------------------------------------

    if is_complete(cache):

        log(
            "[OK] Historical weather dataset "
            "is COMPLETE and VALID."
        )

        log(
            "[CACHE] 0 API requests required."
        )

        return cache

    # --------------------------------------------------------
    # FIND MISSING DATES
    # --------------------------------------------------------

    missing_map = {}

    total_missing = 0

    for district in DISTRICTS:

        missing = get_missing_dates(
            cache,
            district,
        )

        if len(missing) == 0:

            log(
                f"[SKIP] {district}: complete."
            )

        else:

            missing_map[
                district
            ] = missing

            total_missing += len(
                missing
            )

            log(
                f"[MISSING] {district}: "
                f"{len(missing):,} dates"
            )

    if not missing_map:

        log(
            "[OK] No missing weather data."
        )

        return cache

    log(
        f"[INFO] Missing weather dates: "
        f"{total_missing:,}"
    )

    # --------------------------------------------------------
    # PROCESS IN SMALL DISTRICT BATCHES
    # --------------------------------------------------------

    districts = list(
        DISTRICTS
    )

    for batch_start in range(
        0,
        len(districts),
        DISTRICT_BATCH_SIZE,
    ):

        batch = [
            district
            for district in districts[
                batch_start:
                batch_start +
                DISTRICT_BATCH_SIZE
            ]
            if district in missing_map
        ]

        if not batch:
            continue

        # ----------------------------------------------------
        # UNION MISSING DATES
        # ----------------------------------------------------

        union_dates = sorted(
            set(
                date
                for district in batch
                for date in missing_map[
                    district
                ]
            )
        )

        # ----------------------------------------------------
        # CONTIGUOUS RANGES
        # ----------------------------------------------------

        for range_start, range_end in (
            contiguous_ranges(
                union_dates
            )
        ):

            current = range_start

            # Avoid huge requests.
            while current <= range_end:

                chunk_end = min(
                    current
                    + pd.Timedelta(
                        days=364
                    ),
                    range_end,
                )

                chunk_dates = set(
                    pd.date_range(
                        current,
                        chunk_end,
                        freq="D",
                    )
                )

                required_districts = []

                for district in batch:

                    missing_dates = set(
                        missing_map[
                            district
                        ]
                    )

                    if (
                        missing_dates
                        &
                        chunk_dates
                    ):

                        required_districts.append(
                            district
                        )

                if required_districts:

                    new_data = request_batch(
                        required_districts,
                        current,
                        chunk_end,
                    )

                    if not new_data.empty:

                        cache = merge_into_cache(
                            cache,
                            new_data,
                        )

                        # Save immediately.
                        save_cache(
                            cache
                        )

                        # Refresh missing dates.
                        for district in (
                            required_districts
                        ):

                            missing_map[
                                district
                            ] = get_missing_dates(
                                cache,
                                district,
                            )

                    else:

                        log(
                            "[WARNING] No weather "
                            "data returned for "
                            f"{current.date()} -> "
                            f"{chunk_end.date()}"
                        )

                current = (
                    chunk_end
                    +
                    pd.Timedelta(
                        days=1
                    )
                )

                # Deliberate pause between requests.
                time.sleep(2)

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("FINAL WEATHER CACHE CHECK")
    log("=" * 70)

    if is_complete(cache):

        log(
            "[OK] Historical weather is COMPLETE."
        )

        log(
            "[CACHE] Future runs will make "
            "0 historical API requests."
        )

    else:

        log(
            "[WARNING] Historical weather "
            "remains incomplete."
        )

        for district in DISTRICTS:

            remaining = get_missing_dates(
                cache,
                district,
            )

            if len(remaining) > 0:

                log(
                    f"  {district}: "
                    f"{len(remaining):,} missing dates"
                )

    return cache


# ============================================================
# FORECAST CACHE VALIDATION
# ============================================================

def forecast_cache_is_valid():

    if not FORECAST_FILE.exists():
        return False

    try:

        df = pd.read_csv(
            FORECAST_FILE
        )

        df = clean_duplicate_columns(
            df
        )

        if (
            "district" not in df.columns
            or "date" not in df.columns
        ):

            return False

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        today = pd.Timestamp.now(
            tz="Asia/Kolkata"
        ).tz_localize(
            None
        ).normalize()

        expected = set(
            pd.date_range(
                today,
                today
                + pd.Timedelta(
                    days=4
                ),
                freq="D",
            )
        )

        for district in DISTRICTS:

            actual = set(
                df.loc[
                    df[
                        "district"
                    ].map(
                        normalize_district
                    )
                    ==
                    normalize_district(
                        district
                    ),
                    "date",
                ]
                .dropna()
                .dt.normalize()
            )

            if not expected.issubset(
                actual
            ):

                return False

        return True

    except Exception:

        return False


# ============================================================
# FORECAST
# ============================================================

def fetch_forecast(force_refresh=False):

    log("")
    log("=" * 70)
    log("5-DAY FORECAST")
    log("=" * 70)

    # --------------------------------------------------------
    # USE TODAY'S CACHE ONLY WHEN A LIVE REFRESH WAS NOT REQUESTED.
    # The dashboard can pass force_refresh=True to guarantee that it
    # asks Open-Meteo for the newest available forecast.
    # --------------------------------------------------------

    if not force_refresh and forecast_cache_is_valid():

        log(
            "[CACHE] Forecast is COMPLETE."
        )

        log(
            "[CACHE] 0 forecast API requests required."
        )

        return pd.read_csv(
            FORECAST_FILE
        )

    # --------------------------------------------------------
    # BATCH REQUEST
    # --------------------------------------------------------

    frames = []

    districts = list(
        DISTRICTS
    )

    for batch_start in range(
        0,
        len(districts),
        DISTRICT_BATCH_SIZE,
    ):

        batch = districts[
            batch_start:
            batch_start +
            DISTRICT_BATCH_SIZE
        ]

        coordinates = []
        valid_districts = []

        for district in batch:

            coords = get_coordinates(
                district
            )

            if coords is None:
                continue

            coordinates.append(
                coords
            )

            valid_districts.append(
                district
            )

        if not valid_districts:
            continue

        params = {

            "latitude":
                ",".join(
                    str(x[0])
                    for x in coordinates
                ),

            "longitude":
                ",".join(
                    str(x[1])
                    for x in coordinates
                ),

            "forecast_days": 5,

            "daily":
                ",".join(
                    DAILY_VARIABLES
                ),

            "timezone":
                "Asia/Kolkata",

            "temperature_unit":
                "celsius",

            "wind_speed_unit":
                "kmh",

            "precipitation_unit":
                "mm",
        }

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            try:

                log(
                    f"[API] Forecast request: "
                    f"{len(valid_districts)} districts"
                )

                response = requests.get(
                    FORECAST_URL,
                    params=params,
                    timeout=TIMEOUT,
                )

                if response.status_code == 429:

                    retry_after = (
                        response.headers.get(
                            "Retry-After"
                        )
                    )

                    wait = (
                        int(retry_after)
                        if retry_after
                        else 60 * attempt
                    )

                    log(
                        f"[RATE LIMIT] Waiting "
                        f"{wait} seconds."
                    )

                    time.sleep(
                        wait
                    )

                    continue

                response.raise_for_status()

                payload = response.json()

                locations = (
                    [payload]
                    if isinstance(
                        payload,
                        dict
                    )
                    else payload
                )

                for index, location in enumerate(
                    locations
                ):

                    if index >= len(
                        valid_districts
                    ):
                        break

                    daily = location.get(
                        "daily"
                    )

                    if not daily:
                        continue

                    frame = pd.DataFrame(
                        {
                            "date":
                                pd.to_datetime(
                                    daily.get(
                                        "time",
                                        []
                                    ),
                                    errors="coerce",
                                ),

                            "district":
                                valid_districts[
                                    index
                                ],
                        }
                    )

                    for variable in DAILY_VARIABLES:

                        frame[
                            variable
                        ] = daily.get(
                            variable
                        )

                    frames.append(
                        frame
                    )

                break

            except Exception as exc:

                log(
                    f"[WARNING] Forecast request "
                    f"failed "
                    f"(attempt {attempt}/"
                    f"{MAX_RETRIES}): "
                    f"{exc}"
                )

                if attempt < MAX_RETRIES:

                    time.sleep(
                        RETRY_DELAY
                        * attempt
                    )

        time.sleep(2)

    if not frames:

        log(
            "[WARNING] Forecast data unavailable."
        )

        return pd.DataFrame()

    forecast = pd.concat(
        frames,
        ignore_index=True,
    )

    forecast = clean_duplicate_columns(
        forecast
    )

    forecast["date"] = pd.to_datetime(
        forecast["date"],
        errors="coerce",
    )

    forecast = forecast.dropna(
        subset=[
            "date",
            "district",
        ]
    )

    forecast = forecast.drop_duplicates(
        subset=[
            "district",
            "date",
        ],
        keep="last",
    )

    forecast["fetched_at_ist"] = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()

    forecast.to_csv(
        FORECAST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    log(
        f"[OK] Forecast saved: "
        f"{len(forecast):,} rows"
    )

    return forecast
