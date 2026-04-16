#Composite Pollution Index
df["Pollution_Index"] = (
    df["PM2.5"] * 0.4 +
    df["PM10"]  * 0.3 +
    df["NO2"]   * 0.1 +
    df["SO2"]   * 0.1 +
    df["CO"]    * 0.1
)

print("Pollution_Index stats:")
print(df["Pollution_Index"].describe())

# Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Matrix (All Numeric Features)")
plt.tight_layout()
plt.show()

#Feature Correlation with AQI
corr_aqi = df.corr(numeric_only=True)["AQIValue"].drop("AQIValue").sort_values()

plt.figure(figsize=(8, 6))
colors = ["#d73027" if v > 0 else "#4575b4" for v in corr_aqi]
corr_aqi.plot(kind="barh", color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with AQI Value")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.show()

#AQI Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["AQIValue"], bins=40, kde=True, color="steelblue")
plt.title("Distribution of AQI Values")
plt.xlabel("AQI Value")
plt.tight_layout()
plt.show()

print(df["AQIValue"].describe())

#Target Variable Creation
# Outlier clipping before binning (IQR method on raw AQI)
for col in ["PM2.5", "PM10", "NO2", "AQIValue"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

print("AQI range after clipping:", df["AQIValue"].min(), "–", df["AQIValue"].max())

# Create Pollution Zone using AQI percentiles for robust binning
low_cut  = df["AQIValue"].quantile(0.33)
high_cut = df["AQIValue"].quantile(0.66)

print(f"\nBin thresholds — Low < {low_cut:.1f} | Medium: {low_cut:.1f}–{high_cut:.1f} | High > {high_cut:.1f}")

df["Pollution_Zone"] = pd.cut(
    df["AQIValue"],
    bins=[df["AQIValue"].min() - 1, low_cut, high_cut, df["AQIValue"].max()],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

print("\nClass distribution:")
print(df["Pollution_Zone"].value_counts())

# Plot class balance
plt.figure(figsize=(6, 4))
df["Pollution_Zone"].value_counts().plot(kind="bar", color=["#2ecc71", "#f39c12", "#e74c3c"])
plt.title("Pollution Zone Class Distribution")
plt.xlabel("Zone")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Drop Low-Impact & Redundant Features
# Drop: encoded pollutant (low AQI correlation), raw text column, CO and SO2 (low variance/correlation)
cols_to_drop = ["ProminentPollutant_encoded", "ProminentPollutant", "CO", "SO2", "Date"]
df = df.drop(cols_to_drop, axis=1, errors="ignore")

print("Remaining columns:", list(df.columns))

# Define Features & Target
features = ["PM2.5", "PM10", "NO2", "Pollution_Index", "lat", "long", "month"]

  # Keep only features that exist in df
features = [f for f in features if f in df.columns]
print("Selected features:", features)

  # Drop rows with any NaN in features or target
df_clean = df.dropna(subset=features + ["Pollution_Zone"]).copy()
print(f"\nRows before cleaning: {len(df)} | After cleaning: {len(df_clean)}")

X = df_clean[features]
y = df_clean["Pollution_Zone"]

#Train-Test Split (before scaling to prevent data leakage)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print("\nClass balance in train:")
print(y_train.value_counts())

# Feature Scaling (fit on train, transform both)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fit ONLY on training data — transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert back to DataFrames for readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=features)

print("Scaling complete. Feature ranges (train):")
print(X_train_scaled.describe().loc[["min", "max"]])