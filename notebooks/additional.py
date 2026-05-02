import pandas as pd
import numpy as np
from pathlib import Path

#config
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR.parent / "DATA").resolve()
WEATHER_PATH = DATA_DIR / "weather.csv"
HEALTH_PATH = DATA_DIR / "IHME.csv"
URBAN_PATH = DATA_DIR / "urban.csv"

PCT_HIGH = 95
PCT_LOW = 5
MIN_RUN_INCOME = 5

ISO_LOOKUP = {
    "Ukraine": "UKR", "Bosnia and Herz.": "BIH", "Germany": "DEU",
    "Austria": "AUT", "Switzerland": "CHE", "Netherlands": "NLD",
    "Denmark": "DNK", "Sweden": "SWE", "Finland": "FIN",
    "Pakistan": "PAK", "Sri Lanka": "LKA", "Thailand": "THA",
    "Philippines": "PHL", "Japan": "JPN", "Canada": "CAN",
    "Australia": "AUS", "Brazil": "BRA", "Bolivia": "BOL",
    "Colombia": "COL", "Cuba": "CUB", "Honduras": "HND",
    "Nicaragua": "NIC", "Zimbabwe": "ZWE", "Malawi": "MWI",
    "Niger": "NER", "Burundi": "BDI", "Chad": "TCD",
    "Central African Rep.": "CAF", "Mozambique": "MOZ",
    "Sierra Leone": "SLE", "Madagascar": "MDG", "Uganda": "UGA",
    "Gabon": "GAB", "South Africa": "ZAF", "Botswana": "BWA",
    "Algeria": "DZA", "Morocco": "MAR", "Tunisia": "TUN",
    "Egypt": "EGY", "United States of America": "USA",
    "Kazakhstan": "KAZ",
}

VAR_DIRECTIONS = {
    # Heat stress
    "mean_temp": "BOTH", "max_temp": "HIGH", "anomaly_heat_days": "HIGH",
    "total_tropical_nights": "HIGH", "extreme_area": "HIGH",
    # Cold stress
    "anomaly_cold_days": "HIGH", "annual_freeze_burden": "HIGH",
    "min_temp": "LOW", "mean_cold_area": "HIGH",
    # Precipitation / Drought / Runoff
    "heavy_rain_days": "HIGH", "annual_dry_area": "HIGH",
    "drought_episodes": "HIGH", "runoff_days": "HIGH",
    # Air quality
    "pm2.5_mean": "HIGH", "ozone_mean": "HIGH", "no2_mean": "HIGH",
    # Energy / Other
    "annual_total_evap": "BOTH", "mean_annual_net_solar": "BOTH"
    # ... (add others from your list)
}

CAUSES = [
    'Cardiovascular_diseases',
    'Chronic_respiratory_diseases',
    'Environmental_heat_and_cold_exposure'
]

#helper functions
def smooth_income_strict(series, min_run=MIN_RUN_INCOME):
    """Fills short-term income level fluctuations based on surrounding values."""
    s = series.copy().reset_index(drop=True)
    out = s.copy()
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        run_length = j - i
        if run_length < min_run:
            left_val = s[i-1] if i > 0 else None
            right_val = s[j] if j < len(s) else None
            fill_val = left_val if left_val == right_val else left_val or right_val
            out[i:j] = fill_val
        i = j
    return out.values

#data loading
# Load Weather
weather_raw = pd.read_csv(WEATHER_PATH, index_col=0)
weather_df = weather_raw[weather_raw['year'] <= 2023].copy()

# Load Health
health_raw = pd.read_csv(HEALTH_PATH, index_col=0)
health_raw["iso3"] = health_raw["country"].map(ISO_LOOKUP)
health_cleaned = health_raw.drop(columns=['Deaths', 'Incidence'], errors='ignore')

# Pivot Health to Wide Format
health_wide = health_cleaned.pivot_table(
    index=["country", "year", "income_level_mode", "income_level"],
    columns="cause",
    values=["DALYs"],
    aggfunc="first"
).reset_index()

# Flatten Multi-index columns
health_wide.columns = [
    f"{c[0]}_{c[1].replace(' ', '_').replace('&', 'and').replace('/', '_')}" if c[1] else c[0]
    for c in health_wide.columns
]

# Merge Data
df_wide = weather_df.merge(health_wide, on=["country", "year"], how="inner")

# Load Urban Data
urban_df = pd.read_csv(URBAN_PATH, index_col=0)
urban_df = urban_df[urban_df['country'] != 'WORLD']

# Define Feature Keys
H_KEYS = [c for c in df_wide.columns if c.startswith('DALYs_')]
NUM_COLS = df_wide.select_dtypes(include="number").columns.tolist()
C_KEYS = [c for c in NUM_COLS if c not in H_KEYS and c != 'year']

records = []
for country, grp in weather_df.groupby("country"):
    for var, direction in VAR_DIRECTIONS.items():
        if var not in grp.columns: continue
        series = grp[var].dropna()
        if len(series) < 5: continue
        
        records.append({
            "country": country,
            "variable": var,
            "direction": direction,
            "thresh_high": np.percentile(series, PCT_HIGH),
            "thresh_low": np.percentile(series, PCT_LOW),
            "mean": series.mean(),
            "std": series.std(),
            "n_obs": len(series),
        })

percentiles = pd.DataFrame(records).set_index(["country", "variable"]) if records else pd.DataFrame()

extreme_rows = []
if not percentiles.empty:
    for (country, var), thresholds in percentiles.iterrows():
        subset = weather_df[weather_df["country"] == country][["year", var]].dropna()
        
        for _, row in subset.iterrows():
            val = row[var]
            year = int(row["year"])

            is_high = (val >= thresholds["thresh_high"]) and (thresholds["direction"] in ("HIGH", "BOTH"))
            is_low = (val <= thresholds["thresh_low"]) and (thresholds["direction"] in ("LOW", "BOTH"))
            
            direction_label = "HIGH" if is_high else ("LOW" if is_low else "normal")

            extreme_rows.append({
                "country": country,
                "year": year,
                "variable": var,
                "value": val,
                "z_score": (val - thresholds["mean"]) / thresholds["std"] if thresholds["std"] > 0 else 0.0,
                "direction": direction_label,
                "is_extreme": is_high or is_low,
            })

extreme_df = pd.DataFrame(extreme_rows) if extreme_rows else pd.DataFrame()