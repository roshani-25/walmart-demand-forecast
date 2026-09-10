# src/shap_analysis.py

import sqlite3
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# --- Load the saved model and feature list ---
with open('models/random_forest_model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('models/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# --- Load data and recreate the same test set as before ---
conn = sqlite3.connect('data/walmart.db')
df = pd.read_sql_query("SELECT * FROM sales_model_ready", conn)
conn.close()

df['Date'] = pd.to_datetime(df['Date'])
cutoff = df['Date'].max() - pd.Timedelta(weeks=12)
test = df[df['Date'] > cutoff].copy()

X_test = test[feature_cols]

# --- SHAP can be slow on large data, so we sample ---
# 500 rows is enough to see clear overall patterns
X_sample = X_test.sample(500, random_state=42)

print("Computing SHAP values on", len(X_sample), "rows...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# --- Global feature importance: which features matter most overall ---
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig('notebooks/shap_summary.png', dpi=150)
plt.close()
print("Saved: notebooks/shap_summary.png")

# --- Feature importance ranked (mean absolute SHAP value per feature) ---
importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop 10 most important features:")
print(importance.head(10))