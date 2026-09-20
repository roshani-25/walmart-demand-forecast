# src/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Walmart Demand Forecast", layout="wide")
st.title("Walmart Demand Forecasting Dashboard")

# ============================================================
# PIPELINE — runs once on startup, cached after that
# ============================================================

@st.cache_data
def build_dataset():
    train = pd.read_csv('data/raw/train.csv')
    features = pd.read_csv('data/raw/features.csv')
    stores = pd.read_csv('data/raw/stores.csv')

    conn = sqlite3.connect(':memory:')  # in-memory DB, no file written
    train.to_sql('sales', conn, if_exists='replace', index=False)
    features.to_sql('features', conn, if_exists='replace', index=False)
    stores.to_sql('stores', conn, if_exists='replace', index=False)

    query = """
    SELECT 
        s.Store, s.Dept, s.Date, s.Weekly_Sales, s.IsHoliday,
        st.Type, st.Size,
        f.Temperature, f.Fuel_Price, f.CPI, f.Unemployment,
        f.MarkDown1, f.MarkDown2, f.MarkDown3, f.MarkDown4, f.MarkDown5
    FROM sales s
    LEFT JOIN stores st ON s.Store = st.Store
    LEFT JOIN features f ON s.Store = f.Store AND s.Date = f.Date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])

    # --- Cleaning ---
    markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
    df[markdown_cols] = df[markdown_cols].fillna(0)
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # --- Feature engineering ---
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Days_to_Christmas'] = (pd.to_datetime(df['Year'].astype(str) + '-12-25') - df['Date']).dt.days

    superbowl = ['2010-02-12', '2011-02-11', '2012-02-10']
    labor_day = ['2010-09-10', '2011-09-09', '2012-09-07']
    thanksgiving = ['2010-11-26', '2011-11-25', '2012-11-23']
    christmas = ['2010-12-31', '2011-12-30', '2012-12-28']
    date_str = df['Date'].dt.strftime('%Y-%m-%d')
    df['Is_SuperBowl'] = date_str.isin(superbowl).astype(int)
    df['Is_LaborDay'] = date_str.isin(labor_day).astype(int)
    df['Is_Thanksgiving'] = date_str.isin(thanksgiving).astype(int)
    df['Is_Christmas'] = date_str.isin(christmas).astype(int)

    grp = df.groupby(['Store', 'Dept'])['Weekly_Sales']
    df['Lag_1'] = grp.shift(1)
    df['Lag_2'] = grp.shift(2)
    df['Lag_52'] = grp.shift(52)
    df['Rolling_Mean_4'] = grp.transform(lambda x: x.shift(1).rolling(4).mean())
    df['Rolling_Mean_12'] = grp.transform(lambda x: x.shift(1).rolling(12).mean())

    type_dummies = pd.get_dummies(df['Type'], prefix='Type').astype(int)
    df = pd.concat([df, type_dummies], axis=1)
    for col in ['Type_A', 'Type_B', 'Type_C']:
        if col not in df.columns:
            df[col] = 0

    df['Total_Markdown'] = df[markdown_cols].sum(axis=1)

    df_model = df.dropna(subset=['Lag_52']).reset_index(drop=True)
    return df_model

@st.cache_resource
def train_model(df_model):
    feature_cols = [
        'Store', 'Dept', 'Size', 'IsHoliday',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Total_Markdown',
        'Year', 'Month', 'WeekOfYear', 'Days_to_Christmas',
        'Is_SuperBowl', 'Is_LaborDay', 'Is_Thanksgiving', 'Is_Christmas',
        'Lag_1', 'Lag_2', 'Lag_52', 'Rolling_Mean_4', 'Rolling_Mean_12',
        'Type_A', 'Type_B', 'Type_C'
    ]
    cutoff = df_model['Date'].max() - pd.Timedelta(weeks=12)
    train = df_model[df_model['Date'] <= cutoff]

    X_train, y_train = train[feature_cols], train['Weekly_Sales']

    # Smaller forest than the local version — keeps memory/time
    # reasonable on Streamlit Cloud's free tier
    rf = RandomForestRegressor(n_estimators=60, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    return rf, feature_cols

def wmae(dates_df, y_true, y_pred):
    weights = np.where(dates_df['IsHoliday'] == 1, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

# --- Run pipeline (cached — only runs fully on first load) ---
with st.spinner("Loading data and training model (first run only)..."):
    df_model = build_dataset()
    rf, feature_cols = train_model(df_model)

cutoff = df_model['Date'].max() - pd.Timedelta(weeks=12)
test_all = df_model[df_model['Date'] > cutoff].copy()
test_all['Forecast'] = rf.predict(test_all[feature_cols])

wmae_val = wmae(test_all, test_all['Weekly_Sales'], test_all['Forecast'])
rmse_val = np.sqrt(np.mean((test_all['Weekly_Sales'] - test_all['Forecast'])**2))
mae_val = np.mean(np.abs(test_all['Weekly_Sales'] - test_all['Forecast']))

# --- Inventory recommendations ---
LEAD_TIME_WEEKS = 2
Z_SCORE = 1.65

inventory = test_all.groupby(['Store', 'Dept']).agg(
    Avg_Weekly_Forecast=('Forecast', 'mean'),
    Std_Weekly_Forecast=('Forecast', 'std')
).reset_index()
inventory['Std_Weekly_Forecast'] = inventory['Std_Weekly_Forecast'].fillna(0)
inventory['Safety_Stock'] = Z_SCORE * inventory['Std_Weekly_Forecast'] * np.sqrt(LEAD_TIME_WEEKS)
inventory['Reorder_Point'] = (inventory['Avg_Weekly_Forecast'] * LEAD_TIME_WEEKS) + inventory['Safety_Stock']
inventory['Recommended_Reorder_Qty'] = inventory['Avg_Weekly_Forecast'] * LEAD_TIME_WEEKS

# ============================================================
# UI
# ============================================================

st.sidebar.header("Filters")
store_list = sorted(df_model['Store'].unique())
selected_store = st.sidebar.selectbox("Select Store", store_list)
dept_list = sorted(df_model[df_model['Store'] == selected_store]['Dept'].unique())
selected_dept = st.sidebar.selectbox("Select Department", dept_list)

filtered = df_model[(df_model['Store'] == selected_store) & (df_model['Dept'] == selected_dept)].sort_values('Date')
historical = filtered[filtered['Date'] <= cutoff]
test_period = filtered[filtered['Date'] > cutoff].copy()
if len(test_period) > 0:
    test_period['Forecast'] = rf.predict(test_period[feature_cols])

st.subheader(f"Store {selected_store}, Dept {selected_dept} — Sales & Forecast")
chart_data = pd.DataFrame({
    'Date': pd.concat([historical['Date'], test_period['Date']]),
    'Actual': pd.concat([historical['Weekly_Sales'], test_period['Weekly_Sales']]),
}).set_index('Date')
if len(test_period) > 0:
    chart_data['Forecast'] = pd.Series(test_period['Forecast'].values, index=test_period['Date'])
st.line_chart(chart_data)

st.subheader("Inventory Recommendation")
inv_row = inventory[(inventory['Store'] == selected_store) & (inventory['Dept'] == selected_dept)]
if len(inv_row) > 0:
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Weekly Forecast", f"{inv_row['Avg_Weekly_Forecast'].values[0]:,.0f}")
    col2.metric("Safety Stock", f"{inv_row['Safety_Stock'].values[0]:,.0f}")
    col3.metric("Reorder Point", f"{inv_row['Reorder_Point'].values[0]:,.0f}")
    st.write(f"**Recommended Reorder Quantity:** {inv_row['Recommended_Reorder_Qty'].values[0]:,.0f} units")

st.sidebar.header("Model Performance (Test Set)")
st.sidebar.write(f"WMAE: {wmae_val:,.2f}")
st.sidebar.write(f"RMSE: {rmse_val:,.2f}")
st.sidebar.write(f"MAE: {mae_val:,.2f}")

@st.cache_resource
def compute_shap_plot(_rf, _feature_cols, _test_all):
    X_sample = _test_all[_feature_cols].sample(min(300, len(_test_all)), random_state=42)
    explainer = shap.TreeExplainer(_rf)
    shap_values = explainer.shap_values(X_sample)
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    return fig

st.subheader("What Drives These Predictions? (SHAP Summary)")
with st.spinner("Computing SHAP values..."):
    shap_fig = compute_shap_plot(rf, feature_cols, test_all)
    st.pyplot(shap_fig)