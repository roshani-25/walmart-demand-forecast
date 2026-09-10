# src/join_data.py

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

print("Shape:", df.shape)
print(df.head())
print()
print("Nulls per column:")
print(df.isnull().sum())

conn.close()