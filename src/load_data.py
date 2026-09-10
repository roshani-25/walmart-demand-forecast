import pandas as pd
import sqlite3
# 1. Load the CSVs
train = pd.read_csv('data/raw/train.csv')
features = pd.read_csv('data/raw/features.csv')
stores = pd.read_csv('data/raw/stores.csv')

print("train shape:", train.shape)
print("features shape:", features.shape)
print("stores shape:", stores.shape)

# 2. Connect to SQLite database
conn = sqlite3.connect('data/walmart.db')

# 3. Write each dataframe to its own table
train.to_sql('sales', conn, if_exists='replace', index=False)
features.to_sql('features', conn, if_exists='replace', index=False)
stores.to_sql('stores', conn, if_exists='replace', index=False)

# 4. Verify — pull a few rows back from each table
for table in ['sales', 'features', 'stores']:
    df_check = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
    count = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM {table}", conn)
    print(f"\n--- {table} ---")
    print("row count:", count['cnt'][0])
    print(df_check)

conn.close()