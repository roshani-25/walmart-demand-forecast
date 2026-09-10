# src/clean_data.py

import sqlite3
import pandas as pd

conn = sqlite3.connect('data/walmart.db')

query = """
SELECT 
    s.Store,
    s.Dept,
    s.Date,
    s.Weekly_Sales,
    s.IsHoliday,
    st.Type,
    st.Size,
    f.Temperature,
    f.Fuel_Price,
    f.CPI,
    f.Unemployment,
    f.MarkDown1,
    f.MarkDown2,
    f.MarkDown3,
    f.MarkDown4,
    f.MarkDown5
FROM sales s
LEFT JOIN stores st ON s.Store = st.Store
LEFT JOIN features f ON s.Store = f.Store AND s.Date = f.Date
"""

df = pd.read_sql_query(query, conn)
df['Date'] = pd.to_datetime(df['Date'])

# Step 1: Check for duplicates
print("Duplicate rows:", df.duplicated(subset=['Store','Dept','Date']).sum())

# Step 2: Fill MarkDown NaNs with 0 (no promo running = 0, not unknown)
markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
df[markdown_cols] = df[markdown_cols].fillna(0)

# Step 3: Flag negative sales (returns), but keep them
df['Has_Return'] = (df['Weekly_Sales'] < 0).astype(int)
print("Negative sales rows:", df['Has_Return'].sum())

# Step 4: Clean up IsHoliday dtype
df['IsHoliday'] = df['IsHoliday'].astype(int)

# Step 5: Confirm no nulls remain
print("Remaining nulls:")
print(df.isnull().sum())

# Step 6: Save cleaned table back to SQLite
df.to_sql('sales_cleaned', conn, if_exists='replace', index=False)
conn.close()

print("Saved as table: sales_cleaned")