import pandas as pd
import numpy as np
from pathlib import Path

# CONFIG

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR.parent / "DATA").resolve()
WEATHER_PATH = DATA_DIR / "weather.csv"
HEALTH_PATH  = DATA_DIR / "IHME.csv"
URBAN_PATH   = DATA_DIR / "urban.csv"
EXTREMES_PATH = DATA_DIR /"extreme_df.csv"

# CLIMATE VARIABLE METADATA

# Semantic direction for each variable:
#   HIGH  — extreme when unusually high
#   LOW   — extreme when unusually low
#   BOTH  — extreme at either tail
var_directions = {
    # Heat stress
    "mean_temp":                  "BOTH",
    "max_temp":                   "HIGH",
    "anomaly_heat_days":          "HIGH",
    "heat_episodes":              "HIGH",
    "extreme_heat_episodes":      "HIGH",
    "extreme_area":               "HIGH",
    "caution_area":               "HIGH",
    "mean_caution_area":          "HIGH",
    "mean_extreme_area":          "HIGH",
    "max_heat_intensity":         "HIGH",
    "max_rolling_heat_stress":    "HIGH",
    "total_tropical_nights":      "HIGH",
    # Cold stress
    "anomaly_cold_days":          "HIGH",
    "annual_freeze_burden":       "HIGH",
    "min_temp":                   "LOW",
    "winter_mean_snow_density":   "HIGH",
    "mean_cold_area":             "HIGH",
    # Precipitation / Drought
    "heavy_rain_days":            "HIGH",
    "extreme_rain_days":          "HIGH",
    "max_year_precip":            "HIGH",
    "annual_dry_area":            "HIGH",
    "drought_episodes":           "HIGH",
    "mean_dry_area":              "HIGH",
    "evap_deficit":               "HIGH",
    # Runoff / Flood
    "runoff_days":                "HIGH",
    "max_runoff":                 "HIGH",
    # Air quality
    "pm2.5_mean":                 "HIGH",
    "ozone_mean":                 "HIGH",
    "no2_mean":                   "HIGH",
    # Wind / Pressure
    "max_monthly_wind_speed":     "HIGH",
    "mean_annual_wind_speed":     "HIGH",
    "pressure_variability":       "HIGH",
    # Moisture / Evaporation
    "mean_soil_moisture":         "LOW",
    "min_monthly_soil_moisture":  "LOW",
    "annual_total_evap":          "BOTH",
    "annual_potential_evap":      "HIGH",
    "mean_annual_rsn":            "BOTH",
    # Energy fluxes
    "mean_annual_net_solar":      "BOTH",
    "mean_annual_net_thermal":    "BOTH",
    "mean_latent_heat":           "BOTH",
    "mean_sensible_heat":         "BOTH",
}

# Thematic grouping of climate variables (used for analysis / plotting)
CLIMATE_VARS = {
    "Temperature / Heat": [
        "mean_temp", "max_temp", "anomaly_heat_days",
        "heat_episodes", "extreme_heat_episodes",
        "extreme_area", "caution_area",
        "mean_caution_area", "mean_extreme_area",
        "max_heat_intensity", "max_rolling_heat_stress",
        "total_tropical_nights",
    ],
    "Cold / Freeze": [
        "anomaly_cold_days", "annual_freeze_burden",
        "min_temp", "winter_mean_snow_density",
        "mean_cold_area",
    ],
    "Air Pollution": [
        "pm25_mean", "ozone_mean", "no2_mean",
    ],
    "Precipitation / Drought": [
        "heavy_rain_days", "extreme_rain_days",
        "max_year_precip", "annual_dry_area",
        "drought_episodes", "mean_dry_area",
        "evap_deficit",
    ],
    "Runoff / Flood": [
        "runoff_days", "max_runoff",
    ],
    "Wind / Pressure": [
        "max_monthly_wind_speed", "mean_annual_wind_speed",
        "pressure_variability",
    ],
    "Soil / Moisture": [
        "mean_soil_moisture", "min_monthly_soil_moisture",
        "annual_total_evap", "annual_potential_evap",
    ],
    "Energy Fluxes": [
        "mean_annual_net_solar", "mean_annual_net_thermal",
        "mean_latent_heat", "mean_sensible_heat",
    ],
}

CAUSES = [
    'Cardiovascular_diseases',
    'Chronic_respiratory_diseases',
    'Environmental_heat_and_cold_exposure',
]

metrics = ["Deaths", "DALYs", "Incidence"]

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

# DATA LOADING

weather_df = pd.read_csv(WEATHER_PATH, index_col=0)
weather_df = weather_df[weather_df['year'] <= 2023].copy()
health_df = pd.read_csv(HEALTH_PATH, index_col=0)
health_df["iso3"] = health_df["country"].map(ISO_LOOKUP)
urban_df = pd.read_csv(URBAN_PATH, index_col=0)
urban_df = urban_df[urban_df['country'] != 'WORLD']
extreme_df = pd.read_csv(EXTREMES_PATH)

# WIDE-FORMAT DATA (weather + health merged)

health_wide = (
    health_df
    .drop(columns=['Deaths', 'Incidence'], errors='ignore')
    .pivot_table(
        index=["country", "year", "income_level_mode", "income_level"],
        columns="cause",
        values=["DALYs"],
        aggfunc="first"
    )
    .reset_index()
)

health_wide.columns = [
    f"{c[0]}_{c[1].replace(' ', '_').replace('&', 'and').replace('/', '_')}" if c[1] else c[0]
    for c in health_wide.columns
]

df_wide = weather_df.merge(health_wide, on=["country", "year"], how="inner")

# COLUMN GROUPS

climate_features = [c for c in weather_df.columns if c not in ['country', 'year']]

H_KEYS    = [c for c in df_wide.columns if c.startswith('DALYs_')]
NUM_COLS  = df_wide.select_dtypes(include="number").columns.tolist()
C_KEYS    = [c for c in NUM_COLS if c not in H_KEYS and c != 'year']
ECON_KEYS = ['gdp', 'health_exp', 'city', 'rural']
