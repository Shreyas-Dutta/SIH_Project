import re
import sys
import unicodedata
import numpy as np
import pandas as pd
from datetime import datetime

def configure_console():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def log(message=""):
    print(str(message))

def clean_text(value):
    if pd.isna(value): return ""
    value = unicodedata.normalize("NFKC", str(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()

def slugify(value, max_len=80):
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_") or "indicator"
    return value[:max_len]

def safe_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.replace(r"^\s*[-–—]\s*$", "", regex=True).replace({"": np.nan,"NA": np.nan,"N/A": np.nan,"na": np.nan,"n/a": np.nan,"null": np.nan,"None": np.nan}), errors="coerce")

def save_csv(df, filename):
    """Save CSV without letting a locked OneDrive/Excel file abort the pipeline."""
    from ..config import BASE_DIR
    path = BASE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Write through a temporary file first so partially-written outputs are avoided.
        tmp = path.with_name(path.name + ".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        try:
            tmp.replace(path)
        except PermissionError:
            # Windows/OneDrive may keep the destination open. Keep a fresh copy and continue.
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = path.with_name(f"{path.stem}_latest_{stamp}{path.suffix}")
            tmp.replace(fallback)
            log(f"[WARNING] {filename} is locked (likely OneDrive/Excel). Saved fresh copy as {fallback.name}.")
            return fallback
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_latest_{stamp}{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        log(f"[WARNING] {filename} is locked. Saved fresh copy as {fallback.name}.")
        return fallback
    log(f"[OK] Saved {filename}: {len(df):,} rows x {len(df.columns):,} columns")
    return path
