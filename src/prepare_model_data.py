import sqlite3
import pandas as pd

conn = sqlite3.connect('data/walmart.db')
df = pd.read_sql_query("SELECT * FROM sales_features", conn)

before = len(df)

# Drop rows where Lag_52 is null — this also clears out
# Lag_1, Lag_2, and Rolling_Mean_12 nulls automatically,
# since those require LESS history than Lag_52 does
df_model = df.dropna(subset=['Lag_52']).reset_index(drop=True)

after = len(df_model)

print(f"Rows before: {before}")
print(f"Rows after dropping null Lag_52: {after}")
print(f"Rows dropped: {before - after} ({(before - after)/before*100:.1f}%)")
print()

# Confirm no other lag/rolling nulls slipped through
lag_cols = ['Lag_1', 'Lag_2', 'Lag_52', 'Rolling_Mean_4', 'Rolling_Mean_12']
print("Remaining nulls in lag/rolling columns:")
print(df_model[lag_cols].isnull().sum())
print()

df_model['Date'] = pd.to_datetime(df_model['Date'])
print("Date range of final modeling data:", df_model['Date'].min(), "to", df_model['Date'].max())

df_model.to_sql('sales_model_ready', conn, if_exists='replace', index=False)
conn.close()
print("\nSaved as table: sales_model_ready")