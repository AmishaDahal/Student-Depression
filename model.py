#Only depressed value
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

new_filtered_df_new= pd.read_csv("features.csv")
#print(new_filtered_df_new.head())
print(new_filtered_df_new.isnull().sum())



#model training 


# Split the data into training and test sets with shuffling
X = new_filtered_df_new.drop('Depression', axis=1)  # Features
y = new_filtered_df_new['Depression']  # Target

# Shuffling and splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)



# Train a Random Forest model
model = RandomForestClassifier(random_state=42,class_weight='balanced')
model.fit(X_train, y_train)

# Get feature importance
feature_importances = model.feature_importances_

# Create a DataFrame to display feature importance

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


#model accuracy before feature importance
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Performance with Filtered Features:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Reshape feature importance to a 2D array (since heatmap expects a 2D matrix)
importance_matrix = feature_importance_df[['Importance']].T  # Transpose to make it a 2D matrix

# Create the heatmap
plt.figure(figsize=(10, 1))  # Adjust the size as needed
sns.heatmap(importance_matrix, annot=True, cmap='YlGnBu', cbar=True, xticklabels=feature_importance_df['Feature'].values, yticklabels=['Importance'])
plt.title('Feature Importance Heatmap')
plt.legend()
plt.savefig("eda_result/Heatmap.png")


# Remove low-importance features (set a threshold, e.g., < 0.03)
low_importance_features = feature_importance_df[feature_importance_df['Importance'] < 0.03]['Feature'].tolist()
print(f"Low-importance features to drop: {low_importance_features}")

# Drop low-importance features
X_filtered = X.drop(columns=low_importance_features)

print(X_filtered)


# Retrain the model with filtered features
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(
    X_filtered, y, test_size=0.2, random_state=42, shuffle=True
)

model_filtered = RandomForestClassifier(random_state=42,class_weight='balanced')
model_filtered.fit(X_train_filtered, y_train_filtered)


#Evaluate the filtered model

y_pred_filtered = model_filtered.predict(X_test_filtered)

accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
precision = precision_score(y_test_filtered, y_pred_filtered, average='weighted')
recall = recall_score(y_test_filtered, y_pred_filtered, average='weighted')
f1 = f1_score(y_test_filtered, y_pred_filtered, average='weighted')

print("\nModel Performance with Filtered Features:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

#Cross-validation for stability
cv_scores = cross_val_score(model_filtered, X_filtered, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

