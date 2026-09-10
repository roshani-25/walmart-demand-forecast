import sqlite3
import pandas as pd
import numpy as np
import pickle

# --- Load model and feature list ---
with open('models/random_forest_model.pkl', 'rb') as f:
    rf = pickle.load(f)
with open('models/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# --- Load data, recreate test set ---
conn = sqlite3.connect('data/walmart.db')
df = pd.read_sql_query("SELECT * FROM sales_model_ready", conn)
conn.close()

df['Date'] = pd.to_datetime(df['Date'])
cutoff = df['Date'].max() - pd.Timedelta(weeks=12)
test = df[df['Date'] > cutoff].copy()

X_test = test[feature_cols]
test['Forecast'] = rf.predict(X_test)

# --- Inventory assumptions ---
LEAD_TIME_WEEKS = 2
Z_SCORE = 1.65  # ~95% service level

# --- Compute per Store-Dept: average forecast demand + volatility ---
inventory = test.groupby(['Store', 'Dept']).agg(
    Avg_Weekly_Forecast=('Forecast', 'mean'),
    Std_Weekly_Forecast=('Forecast', 'std')
).reset_index()

inventory['Std_Weekly_Forecast'] = inventory['Std_Weekly_Forecast'].fillna(0)

# --- Reorder point and safety stock ---
inventory['Safety_Stock'] = Z_SCORE * inventory['Std_Weekly_Forecast'] * np.sqrt(LEAD_TIME_WEEKS)
inventory['Reorder_Point'] = (inventory['Avg_Weekly_Forecast'] * LEAD_TIME_WEEKS) + inventory['Safety_Stock']
inventory['Recommended_Reorder_Qty'] = inventory['Avg_Weekly_Forecast'] * LEAD_TIME_WEEKS  # simple: cover the lead time gap

inventory = inventory.sort_values('Avg_Weekly_Forecast', ascending=False)

print(inventory.head(10))

# Save to SQLite for the dashboard to use later
conn = sqlite3.connect('data/walmart.db')
inventory.to_sql('inventory_recommendations', conn, if_exists='replace', index=False)
conn.close()

print("\nSaved as table: inventory_recommendations")