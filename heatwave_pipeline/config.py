from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

HMIS_FILE = BASE_DIR / "hmis_data.csv"

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
FORECAST_DAYS = 5

HMIS_RESOURCE_ID = "0c2d45a3-d0b5-4053-bf9a-e00395319472"
HMIS_RESOURCE_PAGE = (
    "https://www.data.gov.in/resource/"
    "item-wise-hmis-report-district-level-assam-upto-april-2019-20"
)

HISTORICAL_CACHE = BASE_DIR / "historical_weather_assam.csv"
PARTIAL_CACHE = BASE_DIR / "historical_weather_partial.csv"
FORECAST_CACHE = BASE_DIR / "forecast_5_day_assam.csv"

# Historical Open-Meteo is configured for ERA5-Land for consistency.
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

DISTRICTS = {
    "Baksa": (26.6986, 91.0650),
    "Barpeta": (26.3225, 91.0060),
    "Bongaigaon": (26.4823, 90.5580),
    "Cachar": (24.8333, 92.7789),
    "Chirang": (26.5736, 90.6127),
    "Darrang": (26.4500, 92.0300),
    "Dhemaji": (27.4800, 94.5800),
    "Dhubri": (26.0200, 89.9700),
    "Dibrugarh": (27.4728, 94.9120),
    "Dima Hasao": (25.5000, 93.0000),
    "Goalpara": (26.1667, 90.6167),
    "Golaghat": (26.5117, 93.9630),
    "Hailakandi": (24.6833, 92.5667),
    "Jorhat": (26.7509, 94.2037),
    "Kamrup M": (26.1445, 91.7362),
    "Kamrup R": (26.0500, 91.5500),
    "Karbi Anglong": (26.0000, 93.5000),
    "Karimganj": (24.8692, 92.3555),
    "Kokrajhar": (26.4000, 90.2700),
    "Lakhimpur": (27.2300, 94.1000),
    "Marigaon": (26.2500, 92.3400),
    "Nagaon": (26.3500, 92.6800),
    "Nalbari": (26.4400, 91.4400),
    "Sibsagar": (26.9840, 94.6380),
    "Sonitpur": (26.6300, 92.8000),
    "Tinsukia": (27.4900, 95.3600),
    "Udalguri": (26.7600, 92.1000),
}

# Compatibility alias used by weather.py
DISTRICT_COORDS = DISTRICTS
