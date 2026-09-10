import sqlite3
import pandas as pd

conn = sqlite3.connect('data/walmart.db')
df = pd.read_sql_query("SELECT * FROM sales_cleaned", conn)
df['Date'] = pd.to_datetime(df['Date'])

# Sort so lag/rolling features are calculated in correct time order
df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

# --- Calendar features ---
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['Days_to_Christmas'] = (pd.to_datetime(df['Year'].astype(str) + '-12-25') - df['Date']).dt.days

# --- Specific holiday flags (exact dates from the competition) ---
superbowl = ['2010-02-12', '2011-02-11', '2012-02-10']
labor_day = ['2010-09-10', '2011-09-09', '2012-09-07']
thanksgiving = ['2010-11-26', '2011-11-25', '2012-11-23']
christmas = ['2010-12-31', '2011-12-30', '2012-12-28']

df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
df['Is_SuperBowl'] = df['Date_str'].isin(superbowl).astype(int)
df['Is_LaborDay'] = df['Date_str'].isin(labor_day).astype(int)
df['Is_Thanksgiving'] = df['Date_str'].isin(thanksgiving).astype(int)
df['Is_Christmas'] = df['Date_str'].isin(christmas).astype(int)
df.drop(columns=['Date_str'], inplace=True)

# --- Lag features (per Store-Dept time series) ---
grp = df.groupby(['Store', 'Dept'])['Weekly_Sales']
df['Lag_1'] = grp.shift(1)
df['Lag_2'] = grp.shift(2)
df['Lag_52'] = grp.shift(52)  # same week, last year

# --- Rolling features (also shifted, so "today" isn't included in its own average) ---
df['Rolling_Mean_4'] = grp.transform(lambda x: x.shift(1).rolling(4).mean())
df['Rolling_Mean_12'] = grp.transform(lambda x: x.shift(1).rolling(12).mean())

# --- One-hot encode Store Type ---
type_dummies = pd.get_dummies(df['Type'], prefix='Type').astype(int)
df = pd.concat([df, type_dummies], axis=1)

# --- Total markdown (simpler single signal than 5 sparse columns) ---
markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
df['Total_Markdown'] = df[markdown_cols].sum(axis=1)

print("Shape after feature engineering:", df.shape)
print()
print("Nulls in new lag/rolling columns (expected — early weeks have no history):")
print(df[['Lag_1','Lag_2','Lag_52','Rolling_Mean_4','Rolling_Mean_12']].isnull().sum())

df.to_sql('sales_features', conn, if_exists='replace', index=False)
conn.close()
print("\nSaved as table: sales_features")