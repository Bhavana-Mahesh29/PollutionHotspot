from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5)
}

results      = {}
trained_models = {}

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Cross-validation (5-fold) on training set
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1_weighted")

    results[name] = {
        "Accuracy":      accuracy_score(y_test, y_pred),
        "Precision":     precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall":        recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score":      f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "CV F1 (mean)": cv_scores.mean(),
        "CV F1 (std)":  cv_scores.std()
    }

    trained_models[name] = (model, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 8.1 Performance Summary Table
    results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values("F1 Score", ascending=False)

print("Model Performance Summary (sorted by F1 Score):")
display(results_df.style.highlight_max(axis=0, color="#c8f7c5").highlight_min(axis=0, color="#f7c5c5"))

# 8.2 Bar Chart Comparison
plot_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]

ax = results_df[plot_cols].plot(
    kind="bar", figsize=(11, 6),
    colormap="Set2", edgecolor="black", linewidth=0.5
)
plt.title("Baseline Model Comparison — Test Set Metrics", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 8.3 Cross-Validation F1 Comparison
fig, ax = plt.subplots(figsize=(8, 5))

model_names = list(results_df.index)
cv_means    = results_df["CV F1 (mean)"].values
cv_stds     = results_df["CV F1 (std)"].values

colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

ax.barh(model_names, cv_means, xerr=cv_stds,
        color=colors[:len(model_names)], edgecolor="black",
        capsize=5, linewidth=0.8)
ax.set_xlabel("Weighted F1 Score (5-Fold CV)")
ax.set_title("Cross-Validation F1 Score with Std Dev")
ax.set_xlim(0, 1.0)
plt.tight_layout()
plt.show()

# 8.4 Confusion Matrices for All Models
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

class_labels = ["Low", "Medium", "High"]

for i, (name, (model, y_pred)) in enumerate(trained_models.items()):
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(f"{name}", fontsize=12, fontweight="bold")

plt.suptitle("Confusion Matrices — All Baseline Models", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

#8.5 Best Model Summary
best_model_name = results_df["F1 Score"].idxmax()
best_score      = results_df.loc[best_model_name, "F1 Score"]

print(f" Best Model: {best_model_name}")
print(f"   Weighted F1 Score: {best_score:.4f}")
print()
print("Full metrics:")
print(results_df.loc[best_model_name])

