# Compute Location Score on the full clean dataset (raw lat/long, not scaled)
df_clean = df_clean.copy()
df_clean["Location_Score"] = df_clean["lat"] + df_clean["long"]
location_median = df_clean["Location_Score"].median()

def recommend_green_solution(row):
    zone = row["Pollution_Zone"]
    loc  = row["Location_Score"]

    if zone == "High":
        if loc > location_median:
            return "Green Walls + Vertical Gardens"
        else:
            return "Urban Forest + Dense Plantation"
    elif zone == "Medium":
        return "Roadside Trees + Green Roofs"
    else:  # Low
        return "Maintain Existing Greenery"

df_clean["Recommendation"] = df_clean.apply(recommend_green_solution, axis=1)

print("Recommendation Distribution:")
print(df_clean["Recommendation"].value_counts())

# 9.1 Recommendation Distribution Plot
palette = {
    "Green Walls + Vertical Gardens":   "#8e44ad",
    "Urban Forest + Dense Plantation":  "#27ae60",
    "Roadside Trees + Green Roofs":     "#f39c12",
    "Maintain Existing Greenery":       "#2ecc71"
}

rec_counts = df_clean["Recommendation"].value_counts()
colors = [palette.get(r, "#95a5a6") for r in rec_counts.index]

plt.figure(figsize=(9, 5))
rec_counts.plot(kind="barh", color=colors, edgecolor="black")
plt.title(" Green Infrastructure Recommendations by Area Count", fontsize=13)
plt.xlabel("Number of Areas")
plt.tight_layout()
plt.show()

# 9.2 Sample Output — Pollution Zones & Recommendations
output_cols = ["AQIValue", "PM2.5", "PM10", "NO2", "Pollution_Zone", "Recommendation"]
available   = [c for c in output_cols if c in df_clean.columns]

display(df_clean[available].head(15))

# 9.3 Save Final Results to Excel
df_clean[available].to_excel("pollution_hotspots_recommendations.xlsx", index=False)
print("✅ Results saved to 'pollution_hotspots_recommendations.xlsx'")
from sklearn.preprocessing import LabelEncoder

le_target = LabelEncoder()
y_train_enc = le_target.fit_transform(y_train)
y_test_enc  = le_target.transform(y_test)

print(le_target.classes_)  # ['High', 'Low', 'Medium'] (order may vary)