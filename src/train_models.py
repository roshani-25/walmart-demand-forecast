# src/train_models.py

import sqlite3
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

conn = sqlite3.connect('data/walmart.db')
df = pd.read_sql_query("SELECT * FROM sales_model_ready", conn)
conn.close()

df['Date'] = pd.to_datetime(df['Date'])

# --- Time-based split: last 12 weeks = test set ---
cutoff = df['Date'].max() - pd.Timedelta(weeks=12)
train = df[df['Date'] <= cutoff].copy()
test = df[df['Date'] > cutoff].copy()

print(f"Train rows: {len(train)}")
print(f"Test rows: {len(test)}")

# --- Feature and target selection ---
feature_cols = [
    'Store', 'Dept', 'Size', 'IsHoliday',
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Total_Markdown',
    'Year', 'Month', 'WeekOfYear', 'Days_to_Christmas',
    'Is_SuperBowl', 'Is_LaborDay', 'Is_Thanksgiving', 'Is_Christmas',
    'Lag_1', 'Lag_2', 'Lag_52', 'Rolling_Mean_4', 'Rolling_Mean_12',
    'Type_A', 'Type_B', 'Type_C'
]
target = 'Weekly_Sales'

X_train, y_train = train[feature_cols], train[target]
X_test, y_test = test[feature_cols], test[target]

# --- WMAE: the competition's actual evaluation metric ---
# Holiday weeks get 5x weight since forecasting errors there are costlier
def wmae(dates_df, y_true, y_pred):
    weights = np.where(dates_df['IsHoliday'] == 1, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

# --- Baseline: Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("\n=== Linear Regression ===")
print("WMAE:", wmae(test, y_test, pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
print("MAE:", mean_absolute_error(y_test, pred_lr))

# --- Random Forest ---
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("\n=== Random Forest ===")
print("WMAE:", wmae(test, y_test, pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))
print("MAE:", mean_absolute_error(y_test, pred_rf))

# --- XGBoost ---
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)

print("\n=== XGBoost ===")
print("WMAE:", wmae(test, y_test, pred_xgb))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_xgb)))
print("MAE:", mean_absolute_error(y_test, pred_xgb))

# --- Save the final chosen model (Random Forest) ---
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Also save the feature column list — needed later to ensure
# any new data fed to the model matches the exact same columns/order
with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("\nModel saved to models/random_forest_model.pkl")