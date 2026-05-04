# Climate-Health Associations Analysis

This repository contains the analysis code accompanying the thesis:

> **Uliana Matvisiv. (2026). *"Analysis of Climate–Health Associations Across Countries Using Statistical Methods and Machine Learning"***

A multi-method analysis of associations between 43 climate indicators and the burden of climate-sensitive diseases (cardiovascular, respiratory, and heat/cold-related) across 40 countries from 1990 to 2023. The study distinguishes structural cross-country differences from dynamic within-country climate effects, with a dedicated case study on Ukraine. 


## Repository Structure

```
├── DATA/                             # All datasets used in this study
├── notebooks/
|   ├── additional.py                 # Data reading/transformation, main variables used in other blocks
│   ├── trends.ipynb                  # Global and country-level DALY trend analysis
│   ├── clustering.ipynb              # Country clustering by climate profiles and residual analysis
│   ├── climate_trends.ipynb          # Climate variables trend and extremes analysis
│   └── variables_relationships.ipynb # Correlations between climate, health, economic, two way fixed effects modelling
├── results/
│   ├── clusters_plots                # Output figures produced in clustering block
│   ├── relations_plots               # Output figures produced in variables relationships block
│   └── trend_plots
│   │   ├── climate_plots             # Output figures produced in climate trends block
│   │   └── health_plots              # Output figures produced in health trends block
└── README.md
```

## Modules Overview

### `additional` — Data Preparation

The auxiliary module imported by all notebooks ( `from additional import…`).
Loads and merges the health, climate, urban, and economic datasets into unified dataframes (`health_df`, `weather_df`, `df_wide`, `extreme_df`) and defines all shared constants and column groupings used across notebooks


### `health_trends` — Health Variables Trends
Analyses how DALYs have evolved globally, by country, and by income group from 1990–2023.

Key components:
- Global DALY trends by disease
- Country-level trend classification (increasing / decreasing / not significant)
- Trend heterogeneity by income group
- Correlations with GDP, health expenditure, and urbanization
- Detection of structural breaks and reversed trajectories
- Top countries with fastest improving/worsening health trends
- DALYs structure by income
- Ukraine case study

Key outputs are saved to `results/trend_plots/health_plots`.

### `climate_trends` — Climate Variable Trends
Analyses how climate variables have changed over time and identifies historically extreme years per country.

Key components:
- Composite climate indices by thematic group (Temperature, Cold, Air Pollution, etc.) computation and global trends plotting
- Computation of annual percentage change for each climate index per country
- Extreme years per country using the 5th/95th within-country percentile threshold across 43 climate variables identification
- Analysis of the relationship between the intensity of climate shocks and changes in DALYs.

Key outputs are saved to `results/trend_polts/climate_plots`. `extreme_df.csv` and `extreme_years_table.csv` are saved to `DATA/` for use by other notebooks without re-computation

### `variables_relationships` — Variable Relationships
Quantifies the relationship between climate variables and DALYs at multiple levels of analysis.

Key components:
- Pearson and Spearman correlation matrices between climate variables and DALYs outcomes at three levels: Pooled (no fixed effects), Within-country (country-demeaned),Two-way fixed effects (country + year demeaned)
- Corellations of DALYs and climate by income and comparison
- Distributed lag panel models to test delayed climate effects
- One/two way fixed effects models for estimating dynamic effects of climate variables on health outcomes in panel data
- Ridge regression and XGBoost models with Leave-One-Country-Out (LOCO) cross-validation to assess out-of-sample predictive power
- SHAP values and permutation importance to rank the most influential climate variables per disease

Key outputs are saved to `results/relations_plots`. 

### `clustering` — Country Clustering
Groups countries by their climate profile and evaluates where each sits relative to its peers. Cluster assignments are used throughout the trend analysis as a control variable.

Key components:
- Country-level profile for 40 countries composition
- Testing three feature sets: all climate + DALYs, health-relevant climate subset, and PCA-reduced (85% variance)
- Optimal k selection via silhouette score across k = 2–8 for both K-means and Gaussian Mixture Model (GMM) clustering
- Cluster agreement validation between methods using Adjusted Rand Index (ARI)
- Performs residual analysis to flag countries whose burden is unexpectedly high or low for their climate cluster and income level.

Key outputs are saved to `results/clusters_plots`. Also clustering csv is saved in `DATA/` for future use.

## Data Sources

- **Health data:** IHME Global Burden of Disease (GBD) — DALYs, deaths, incidence by cause and country
- **Climate data:** ERA5 reanalysis and State of Global Air database — temperature extremes, heat/cold intensity, precipitation, wind, NO₂, PM2.5, ozone etc.
- **Socioeconomic data:** World Bank — income level, GDP per capita, health expenditure, urbanization rates

Note: The actual process of constructing the datasets from the raw data obtained from the specified sources is not included in the repository and is described only in the thesis itself.

## Setup & Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
plotly
xgboost
shap
```
Python 3.9+ recommended.
