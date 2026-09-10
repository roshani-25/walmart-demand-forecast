import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('data/walmart.db')
df = pd.read_sql_query("SELECT * FROM sales_cleaned", conn)
conn.close()

df['Date'] = pd.to_datetime(df['Date'])

# 1. Overall weekly sales trend
trend = df.groupby('Date')['Weekly_Sales'].mean()
plt.figure(figsize=(12,5))
plt.plot(trend.index, trend.values)
plt.title('Average Weekly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Avg Weekly Sales')
plt.savefig('notebooks/trend.png')
plt.close()

# 2. Holiday vs non-holiday
print("Holiday vs Non-Holiday avg sales:")
print(df.groupby('IsHoliday')['Weekly_Sales'].mean())

# 3. Store type
print("\nAvg sales by Store Type:")
print(df.groupby('Type')['Weekly_Sales'].mean().sort_values(ascending=False))

# 4. Top departments
print("\nTop 10 departments by avg sales:")
print(df.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False).head(10))

# 5. Correlation with external factors
print("\nCorrelation with Weekly_Sales:")
print(df[['Weekly_Sales','Temperature','Fuel_Price','CPI','Unemployment']].corr()['Weekly_Sales'])