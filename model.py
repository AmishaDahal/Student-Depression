#Only depressed value
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 
from sklearn.ensemble import GradientBoostingClassifier

new_filtered_df_new= pd.read_csv("features.csv")
#print(new_filtered_df_new.head())
new_filtered_df_new.shape



#model training 


# Split the data into training and test sets with shuffling
X = new_filtered_df_new.drop('Depression', axis=1)  # Features
y = new_filtered_df_new['Depression']  # Target

# Shuffling and splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Train a Random Forest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Get feature importance
feature_importances = model.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,  # Use X_train.columns to align with training features
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Model accuracy before feature importance
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print metrics
print("\nModel Performance with Random Forest:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Reshape feature importance to a 2D array (since heatmap expects a 2D matrix)
importance_matrix = feature_importance_df[['Importance']].T  # Transpose to make it a 2D matrix

# Create the heatmap
plt.figure(figsize=(15, 10))  # Adjust the size as needed
sns.heatmap(
    importance_matrix,
    annot=True,
    cmap='YlGnBu',
    cbar=True,
    xticklabels=feature_importance_df['Feature'].values,
    yticklabels=['Importance']
)
plt.title('Feature Importance Heatmap')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()

# Save the heatmap
plt.savefig("eda_result/Heatmap.png")


# Cross-validation using StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(X_train_fold, y_train_fold)

    # Predict on the test set
    y_pred_fold = model.predict(X_test_fold)

    # Calculate metrics
    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    precisions.append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    recalls.append(recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))

# Compute mean and standard deviation for each metric
print("\nStratified K-Fold Cross-Validation Random Forest Results:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

##Random Forest classifier
##logistic regression
##decision tree
##XGboost
##gradient boost 


# Train a logistic regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# Use coefficients for feature importance
coefficients = np.abs(logistic_model.coef_[0])  # Get absolute values of coefficients
feature_importances = coefficients / np.sum(coefficients)  # Normalize for interpretability

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot a heatmap for feature importance
plt.figure(figsize=(15, 10))
sns.heatmap(
    feature_importance_df[['Importance']].T, 
    annot=True, 
    cmap='YlGnBu', 
    cbar=True, 
    xticklabels=feature_importance_df['Feature'].values, 
    yticklabels=['Importance']
)
plt.title('Logistic Regression Feature Importance Heatmap')
plt.savefig("eda_result/Logistic_Heatmap.png")

# Evaluate model performance on the holdout test set
log_y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, log_y_pred)
precision = precision_score(y_test, log_y_pred, average='weighted')
recall = recall_score(y_test, log_y_pred, average='weighted')
f1 = f1_score(y_test, log_y_pred, average='weighted')

print("\nModel Performance with Logistic Regression:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Cross-validation using stratified K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Stratified K-fold cross-validation
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Train the model on the fold
    logistic_model.fit(X_train_fold, y_train_fold)

    # Predict on the test set of the fold
    y_pred_fold = logistic_model.predict(X_test_fold)

    # Calculate metrics
    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    precisions.append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    recalls.append(recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))

# Compute mean and standard deviation for each metric
print("\nStratified K-Fold Cross-Validation Logistic Regression Results:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")



#import numpy as np

# Define the model
decision_tree_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Train the model
decision_tree_model.fit(X_train, y_train)

# Predict the model
decision_y_pred = decision_tree_model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, decision_y_pred)
precision = precision_score(y_test, decision_y_pred, average='weighted')
recall = recall_score(y_test, decision_y_pred, average='weighted')
f1 = f1_score(y_test, decision_y_pred, average='weighted')

# Print metrics
print("Decision Tree Model Performance")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot feature importance heatmap
feature_importances = decision_tree_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(15, 10))
sns.heatmap(
    feature_importance_df[['Importance']].T,
    annot=True,
    cmap='YlGnBu',
    cbar=True,
    xticklabels=feature_importance_df['Feature'].values,
    yticklabels=['Importance']
)
plt.title('Decision Tree Feature Importance Heatmap')
plt.savefig("eda_result/Decision_Heatmap.png")

# Cross-validation of decision tree model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []
f1_scores = []

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    decision_tree_model.fit(X_train_fold, y_train_fold)

    # Predict on the test set
    y_pred_fold = decision_tree_model.predict(X_test_fold)

    # Calculate metrics
    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    precisions.append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    recalls.append(recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))

# Compute mean and standard deviation for each metric
print("\nStratified K-Fold Cross-Validation Decision Tree Results:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")


##XG BOOST

# Initialize the XGBoost Classifier
XG_model = XGBClassifier(
    random_state=42, 
    scale_pos_weight=1, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
)

# Fit the model to the training data
XG_model.fit(X_train, y_train)



# Get feature importance
feature_importances = XG_model.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,  # Use X_train.columns to align with training features
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Model accuracy before feature importance
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print metrics
print("\nModel Performance with XGBoost:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Reshape feature importance to a 2D array (since heatmap expects a 2D matrix)
importance_matrix = feature_importance_df[['Importance']].T  # Transpose to make it a 2D matrix

# Create the heatmap
plt.figure(figsize=(15, 10))  # Adjust the size as needed
sns.heatmap(
    importance_matrix,
    annot=True,
    cmap='YlGnBu',
    cbar=True,
    xticklabels=feature_importance_df['Feature'].values,
    yticklabels=['Importance']
)
plt.title('Feature Importance Heatmap')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()

# Save the heatmap
plt.savefig("eda_result/XGBoost_Heatmap.png")


# Cross-validation using StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(X_train_fold, y_train_fold)

    # Predict on the test set
    y_pred_fold = model.predict(X_test_fold)

    # Calculate metrics
    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    precisions.append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    recalls.append(recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))

# Compute mean and standard deviation for each metric
print("\nStratified K-Fold Cross-Validation XGBoost Results:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")



# Train a Gradient Boosting model
gradient_model = GradientBoostingClassifier(random_state=42)
gradient_model.fit(X_train, y_train)

# Get feature importance
feature_importances = gradient_model.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,  # Use X_train.columns to align with training features
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Model accuracy before feature importance
y_pred = gradient_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print metrics
print("\nModel Performance with Gradient Boosting:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Reshape feature importance to a 2D array (since heatmap expects a 2D matrix)
importance_matrix = feature_importance_df[['Importance']].T  # Transpose to make it a 2D matrix

# Create the heatmap
plt.figure(figsize=(15, 10))  # Adjust the size as needed
sns.heatmap(
    importance_matrix,
    annot=True,
    cmap='YlGnBu',
    cbar=True,
    xticklabels=feature_importance_df['Feature'].values,
    yticklabels=['Importance']
)
plt.title('Feature Importance Heatmap')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()

# Save the heatmap
plt.savefig("eda_result/GradientBoosting_Heatmap.png")


# Cross-validation using StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store metrics for each fold
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(X_train_fold, y_train_fold)

    # Predict on the test set
    y_pred_fold = model.predict(X_test_fold)

    # Calculate metrics
    accuracies.append(accuracy_score(y_test_fold, y_pred_fold))
    precisions.append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    recalls.append(recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))
    f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0))

# Compute mean and standard deviation for each metric
print("\nStratified K-Fold Cross-Validation Gradient Boosting Results:")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")


#Consolidate metrics for each model
model_comparison = {
    'Model': ['Random Forest', 'Logistic Regression', 'Decision Tree', 'XGBoost', 'Gradient Boosting'],
    'Accuracy (mean)': [
        f"{np.mean(accuracy):.4f}",
        f"{np.mean(accuracy):.4f} ",
        f"{np.mean(accuracy):.4f} ",
        f"{np.mean(accuracy):.4f}",
        f"{np.mean(accuracy):.4f} ",
    ],
    'Precision (mean)': [
        f"{np.mean(precision):.4f}",
        f"{np.mean(precision):.4f} ",
        f"{np.mean(precision):.4f}",
        f"{np.mean(precision):.4f}",
        f"{np.mean(precision):.4f}",
    ],
    'Recall (mean)': [
        f"{np.mean(recall):.4f} ",
        f"{np.mean(recall):.4f} ",
        f"{np.mean(recall):.4f} ",
        f"{np.mean(recall):.4f}",
        f"{np.mean(recall):.4f} ",
    ],
    'F1 Score (mean)': [
        f"{np.mean(f1):.4f} ",
        f"{np.mean(f1):.4f}",
        f"{np.mean(f1):.4f} ",
        f"{np.mean(f1):.4f}",
        f"{np.mean(f1):.4f} ",
    ],
}

# Create a DataFrame
comparison_df = pd.DataFrame(model_comparison)

# Display the DataFrame
print("\nModel Comparison:")
print(comparison_df)

# Save the comparison table as a CSV file
comparison_df.to_csv("eda_result/model_comparison.csv", index=False)


# Consolidate metrics for each model

cross_val_model_comparison = {
    'Model': ['Random Forest', 'Logistic Regression', 'Decision Tree', 'XGBoost', 'Gradient Boosting'],
    'Accuracy (mean ± std)': [
        f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
        f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
    ],
    'Precision (mean ± std)': [
        f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
    ],
    'Recall (mean ± std)': [
        f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
    ],
    'F1 Score (mean ± std)': [
        f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
        f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
    ],
}

# Create a DataFrame
comparison_df_new = pd.DataFrame(cross_val_model_comparison)

# Display the DataFrame
print("\n Crossvalidation Model Comparision:")
print(comparison_df_new)

# Save the comparison table as a CSV file

comparison_df_new.to_csv("eda_result/cross_validation_model_comparison.csv", index=False)
