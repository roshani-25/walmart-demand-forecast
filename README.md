# Walmart Demand Forecasting & Inventory Recommendation System

**[Live Demo](https://walmart-demand-forecast-pekgwks6gbu6v33unfp2em.streamlit.app/)**

End-to-end machine learning pipeline that forecasts weekly retail demand and converts those forecasts into actionable inventory reorder recommendations.

## Overview

Built on Walmart's store sales dataset (45 stores, department-level weekly sales, Feb 2010 – Oct 2012). The pipeline moves from raw CSVs through a SQL data layer, cleaning, feature engineering, model comparison, and explainability, ending in a deployed interactive dashboard.

## Pipeline

1. **SQL Data Layer** — Raw CSVs loaded into SQLite as three normalized tables (`sales`, `stores`, `features`), joined via SQL rather than flat-file merging
2. **Data Cleaning** — Handled structurally missing markdown data, flagged negative sales (returns), verified no duplicate Store-Dept-Date records
3. **EDA** — Identified seasonality patterns, store-type effects, and department-level variance
4. **Feature Engineering** — Calendar features, per-holiday indicators, lag features (1-week, 2-week, 52-week), rolling averages, one-hot encoded store types
5. **Modeling** — Compared Linear Regression, Random Forest, and XGBoost using a time-based train/test split
6. **Evaluation** — Scored with WMAE (the competition's official metric, weighting holiday weeks 5x), plus RMSE and MAE
7. **Explainability** — SHAP analysis on the selected model
8. **Inventory Logic** — Forecasts converted into safety stock and reorder points, scaled per Store-Dept by forecast volatility
9. **Deployment** — Streamlit dashboard hosted on Streamlit Community Cloud

## Results

| Model | WMAE | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 1559.79 | 2976.65 | 1521.22 |
| Random Forest | **1258.91** | **2592.58** | 1227.60 |
| XGBoost | 1268.25 | 2619.02 | **1225.12** |

Random Forest was selected as the final model — it led on WMAE and RMSE, with XGBoost essentially tied (within ~1%).

## Key Findings

- **Holiday effects are highly uneven.** A single `IsHoliday` flag showed only a ~7% average sales lift, which masked the real pattern: Thanksgiving week drives a large spike while Super Bowl and Labor Day are nearly flat. Separate per-holiday features were engineered as a result.
- **Lag features dominate.** SHAP analysis showed `Lag_52` (same week last year) and `Lag_1` (previous week) far outweigh all other features — consistent with the strong year-over-year seasonality seen in EDA.
- **Department identity is absorbed by lags.** Departments showed large raw sales variance in EDA, but low SHAP importance — because a department's typical sales level is already encoded in its own lag values.
- **Macro-economic factors showed negligible correlation.** Temperature, fuel price, CPI, and unemployment all correlated near zero with weekly sales in this dataset.

## Design Decisions

- **Time-based split, not random** — random splitting would leak future information into training, which is unrealistic for forecasting.
- **Dropped the first year of data** (~38% of rows) to enable `Lag_52`, trading training volume for a strong year-over-year feature. SHAP confirmed this was worth it.
- **Volatility-scaled safety stock** — buffer sizing is per Store-Dept based on forecast standard deviation, rather than a flat buffer that would over-stock stable products and under-stock volatile ones.

## Tech Stack

Python · SQLite · pandas · scikit-learn · XGBoost · SHAP · Streamlit · matplotlib

## Assumptions

Inventory calculations assume a 2-week supplier lead time and a 95% service level (Z = 1.65). Real deployments would source these from actual supplier data.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

The app rebuilds the dataset and trains the model on first launch (results are cached afterward).
