##Data Loading & Inspection
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("/content/MLfeb8.xlsx")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
display(df.head())
print("\nData Types & Null Counts:")
df.info()

## Convert Pollutant Columns to Numeric
pollutant_cols = ["SO2", "NO2", "CO", "PM2.5", "PM10", "AQIValue"]

for col in pollutant_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Null counts after conversion:")
print(df[pollutant_cols].isnull().sum())

#3Handle Missing Values (Mean Imputation)
for col in pollutant_cols:
    df[col] = df[col].fillna(df[col].mean())

print("Null counts after imputation:")
print(df[pollutant_cols].isnull().sum())

# Parse Dates & Extract Date Features
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

df["year"]    = df["Date"].dt.year
df["month"]   = df["Date"].dt.month
df["day"]     = df["Date"].dt.day
df["weekday"] = df["Date"].dt.weekday

# Drop year if it has no variation (single-year datasets add no information)
if df["year"].nunique() == 1:
    df = df.drop("year", axis=1)
    print("'year' dropped — only one unique value found.")

print(df[["month", "day", "weekday"]].head())

# Label Encoding for Prominent Pollutant
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
if "ProminentPollutant" in df.columns:
    df["ProminentPollutant_encoded"] = le.fit_transform(df["ProminentPollutant"].astype(str))
    print("ProminentPollutant classes:", le.classes_)

