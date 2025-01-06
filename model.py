import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data_path = "features.csv"
data = pd.read_csv(data_path)

X = data.drop(columns=["Depression"])
y = data["Depression"]

def stratified_kfold_cv(model, X, y, model_name, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics["Recall"].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics["F1 Score"].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    averaged_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    return averaged_metrics

models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
}

all_metrics = {}
feature_importance_dict = {}
n_splits = 5

for name, model in models.items():
    print(f"Evaluating {name}...")
    metrics = stratified_kfold_cv(model, X, y, name, n_splits=n_splits)
    all_metrics[name] = metrics

    model.fit(X, y)
    if hasattr(model, "feature_importances_"):
        feature_importance_dict[name] = model.feature_importances_
    elif hasattr(model, "coef_"):
        feature_importance_dict[name] = np.abs(model.coef_[0])

comparison_df = pd.DataFrame(all_metrics).T
output_csv_path = "stratified_kfold_model_comparison_metrics.csv"
comparison_df.to_csv(output_csv_path)
print(f"\nModel comparison metrics (Stratified K-Fold) saved to: {output_csv_path}")

plt.figure(figsize=(12, 8))
comparison_df.plot(kind='bar', figsize=(12, 8), colormap='viridis', edgecolor='black')
plt.title(f"Model Performance Metrics Comparison ({n_splits}-Fold Stratified CV)", fontsize=16)
plt.ylabel("Metric Value", fontsize=14)
plt.xlabel("Models", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.legend(title="Metrics", loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig("stratified_kfold_model_comparison_metrics_plot.png")


feature_importance_df = pd.DataFrame(feature_importance_dict, index=X.columns)
plt.figure(figsize=(10, 8))
sns.heatmap(feature_importance_df, annot=True, cmap="viridis", cbar=True, fmt=".2f")
plt.title("Feature Importance for Each Model", fontsize=16)
plt.xlabel("Models", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.tight_layout()
feature_importance_path = "feature_importance_heatmap.png"
plt.savefig(feature_importance_path)
print(f"Feature importance heatmap saved to: {feature_importance_path}")
