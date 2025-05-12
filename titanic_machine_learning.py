# Lily Gates
# May 2025

import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# IMPORT DATA FILE
# Note: must be in current working dir
# ----------------------------------------------------------

train_data = pd.read_csv("train.csv")

# Preview data
#train_data.head()
#train_data.info()

############################################################
# ----------------------------------------------------------
# PRE-PROCESS DATASET
# ----------------------------------------------------------
############################################################

# ----------------------------------------------------------
# One-Hot Encoding Categorical Variables
# ----------------------------------------------------------
train_data_encoded = pd.get_dummies(train_data, columns=['Pclass', 'Sex', 'Embarked'], drop_first=False)

# ----------------------------------------------------------
# Drop the specified columns from the DataFrame
# ----------------------------------------------------------
train_data_cleaned = train_data_encoded.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# ----------------------------------------------------------
# Replace NaN values in "Age" with the median of "Age"
# ----------------------------------------------------------

median_age = train_data_cleaned["Age"].median()
train_data_cleaned['Age'] = train_data_cleaned['Age'].fillna(median_age)

# Confirm column names
#train_data_cleaned.columns

# Check for NaNs in the specified columns
columns_to_check = ['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',
       'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q',
       'Embarked_S']

# Get the number of NaNs in each of these columns
nan_counts = train_data_cleaned[columns_to_check].isnull().sum()
#print(nan_counts)

# ----------------------------------------------------------
# EXPLORE DATASET: HISTOGRAM OF AGE DISTRIBUTION
# ----------------------------------------------------------

# Histogram of age distribution, bins are every 5 years
sns.histplot(train_data_cleaned['Age'], kde=True, binwidth=5)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig("histog_age.png", dpi=300, bbox_inches='tight')
print("Histogram of age distribution saved as: 'histog_age.png'")
plt.show()

# ----------------------------------------------------------
# DISPLAY CLEANED DATAFRAME: Ready to Train and Test with
# ----------------------------------------------------------
train_data_cleaned

############################################################
# ----------------------------------------------------------
# SPLIT TEST VS. TRAIN DATA
# ----------------------------------------------------------
############################################################

# Split data into features and target
X = train_data_cleaned.drop("Survived", axis=1)
y = train_data_cleaned["Survived"]

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


############################################################
# ----------------------------------------------------------
# LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------
############################################################

# Logistic Regression model
logreg_model = LogisticRegression(random_state=42, max_iter=400)
logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)

# ----------------------------------------------------------
# VISUALIZE: LOGISTIC REGRESSION MODEL
# ----------------------------------------------------------

# Plotting the feature importance (coefficients)
coefficients = logreg_model.coef_[0]
features = X_train.columns

# Sort the coefficients and features by coefficient values (lowest to highest)
sorted_indices = np.argsort(coefficients)  # Get indices that would sort the coefficients
sorted_coefficients = coefficients[sorted_indices]
sorted_features = features[sorted_indices]

# Define colors based on the coefficient values
colors = ['#35b779' if coef > 0 else '#a52c60' for coef in sorted_coefficients]  # Green for positive, Red for negative

# Adjust the figure size for better label visibility
plt.figure(figsize=(12, 8))  # Increase width and height

# Plot the horizontal bar chart with the sorted coefficients and color logic
plt.barh(sorted_features, sorted_coefficients, color=colors)
plt.title('Logistic Regression Feature Importance (Coefficients)', fontsize=14)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)

# Rotate the feature labels if necessary
plt.yticks(rotation=0)  # This ensures labels are horizontal

# Adjust layout to make sure everything fits
plt.tight_layout()

# Save and show the plot
plt.savefig("logreg_coefficients.png", dpi=300, bbox_inches='tight')
print("Logistic regression coefficient graph saved as: 'logreg_coefficients.png'")
plt.show()


############################################################
# ----------------------------------------------------------
# DECISION TREE
# ----------------------------------------------------------
############################################################

# ----------------------------------------------------------
# Refine the Decision Tree Model: Find Optimal Depth
# ----------------------------------------------------------
# Define a range of depths to test
depths = range(1, 21)  # You can adjust this range if needed
cv_scores = []

# Perform cross-validation for each depth
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold CV
    cv_scores.append(scores.mean())  # Store the mean score

# Plot the results
plt.plot(depths, cv_scores, marker='o', label='Cross-Validation Accuracy')
plt.title('Cross-Validation Scores vs Tree Depth')
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Accuracy')

# Find the max accuracy and corresponding depth
best_depth = depths[cv_scores.index(max(cv_scores))]
best_score = max(cv_scores)

# Mark the best point
plt.scatter(best_depth, best_score, color='red', zorder=5, label=f'Best Depth: {best_depth}\nAccuracy: {best_score:.4f}')

# Add legend
plt.legend()
plt.savefig("cross_val_scores_vs_depth.png", dpi=300, bbox_inches='tight')
print("The cross-validation scores graph is saved as: 'cross_val_scores_vs_depth.png'")
# Show the plot
plt.show()

# Output the optimal max_depth and corresponding score
print(f"The optimal max_depth is {best_depth} with a cross-validation accuracy of {best_score:.4f}")

# ----------------------------------------------------------
# Graph the Decision Tree Model
# ----------------------------------------------------------
# Decision Tree model
decision_tree_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
decision_tree_model.fit(X_train, y_train)
decision_tree_pred = decision_tree_model.predict(X_test)

# Plot the decision tree with optimal depth
plt.figure(figsize=(20, 10))
plot_tree(
    decision_tree_model,
    filled=True,
    feature_names=X_train.columns,
    class_names=["Not Survived", "Survived"],
    rounded=True,
    fontsize=10
)

plt.title(f"Decision Tree with Optimal Depth\n(max_depth={best_depth})", fontsize=16)

# Save the plot to a file
plt.savefig("decision_tree_optimal_depth.png", dpi=300, bbox_inches='tight')
print("Decision tree graph saved as: 'decision_tree_optimal_depth.png'")
# Show the plot
plt.show()

# ----------------------------------------------------------
# Get Rules for the Decision Tree Model
# ----------------------------------------------------------

# Get the rules from the decision tree model
tree_rules = export_text(decision_tree_model, feature_names=list(X_train.columns), max_depth=best_depth)

# Print the rules to the console (optional)
print(tree_rules)

# Save the rules to a text file
with open("decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)

print("The decision tree rules have been saved to 'decision_tree_rules.txt'.")


############################################################
# ----------------------------------------------------------
# RANDOM FOREST MODEL
# ----------------------------------------------------------
############################################################
# Fit the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
random_forest_pred = random_forest_model.predict(X_test)

# Extract feature importances from the trained model
feature_importances = random_forest_model.feature_importances_

# Create a DataFrame with feature names and their importance
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,  # Assuming 'X_train' has the feature names
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# ----------------------------------------------------------
# VISUALIZE: RANDOM FOREST
# ----------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    hue='Feature',         # Assign y variable to hue
    data=feature_importance_df,
    palette='viridis',     # Or your chosen palette
    legend=False           # Disable unnecessary legend
)

plt.title('Random Forest Feature Importance', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png", dpi=300, bbox_inches='tight')
print("Random forest feature importance graph saved as: 'random_forest_feature_importance.png'")
plt.show()

############################################################
# ----------------------------------------------------------
# EVALUATE AND COMPARE ALL MODEL METRICS
# ----------------------------------------------------------
############################################################

# ----------------------------------------------------------
# DATAFRAME TO EVALUATE MODEL METRICS
# ----------------------------------------------------------

# Calculate performance metrics for each model

# For Logistic Regression
logreg_pred = logreg_model.predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_pred)
logreg_prec = precision_score(y_test, logreg_pred)
logreg_recall = recall_score(y_test, logreg_pred)
logreg_f1 = f1_score(y_test, logreg_pred)

# For Decision Tree
dt_pred = decision_tree_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

# For Random Forest
rf_pred = random_forest_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# Ensure that the full DataFrame can be displayed without truncation
pd.set_option('display.max_rows', None)  # Set to None to display all rows
pd.set_option('display.max_columns', None)  # Set to None to display all columns

# Create a dictionary of model metrics
model_metrics = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [logreg_acc, dt_acc, rf_acc],
    "Precision": [logreg_prec, dt_prec, rf_prec],
    "Recall": [logreg_recall, dt_recall, rf_recall],
    "F1 Score": [logreg_f1, dt_f1, rf_f1]
}

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(model_metrics)

# Set the 'Model' column as the index
metrics_df.set_index('Model', inplace=True)

# Display the full DataFrame
print("\n-----------------------------------------\n")
print(metrics_df)
print("\n-----------------------------------------\n")

# ----------------------------------------------------------
# BAR PLOTS: COMPARE MODELS AND EVALUATION METRICS
# ----------------------------------------------------------

# --------- Bar Plot: Models on x-axis ----------
# Set a custom color palette
plasma = ['#0d0887', '#7201a8', '#bd3786', '#ed7953']

ax1 = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(
    kind='bar',
    figsize=(10, 6),
    color=plasma
)
plt.title('Comparison of Evaluation Metrics by Model', fontsize=14)
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0.7, 0.82)
plt.xticks(rotation=0)
plt.legend(title='Metric')  # Legend inside the plot
plt.tight_layout()
plt.savefig("metric_by_model_comparison.png", dpi=300, bbox_inches='tight')
print("Comparison of evaluation metrics saved as: 'metric_by_model_comparison.png'")
plt.show()

# --------- Bar Plot: Metrics on x-axis ----------

viridis = ['#35b779', '#3e4989', '#440154']

ax2 = metrics_df.T.plot(
    kind='bar',
    figsize=(10, 6),
    color=viridis
)
plt.title('Comparison of Models by Evaluation Metric', fontsize=14)
plt.ylabel('Score')
plt.xlabel('Metric')
plt.ylim(0.7, 0.82)
plt.xticks(rotation=0)
plt.legend(title='Model')  # Legend inside the plot
plt.tight_layout()
plt.savefig("model_by_metric_comparison.png", dpi=300, bbox_inches='tight')
print("Comparison of models saved as: 'model_by_metric_comparison.png'")
plt.show()

# ----------------------------------------------------------
# CONFUSION MATRIX FOR ALL MODELS
# ----------------------------------------------------------

# Evaluate and display confusion matrix for Logistic Regression
logreg_cm = confusion_matrix(y_test, logreg_pred)
logreg_disp = ConfusionMatrixDisplay(confusion_matrix=logreg_cm, display_labels=["Not Survived", "Survived"])
logreg_disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("confusion_matrix_logreg.png", dpi=300, bbox_inches='tight')
print("Confusion matrix for logistic regression saved as: 'confusion_matrix_logreg.png'")
plt.show()

# Evaluate and display confusion matrix for Decision Tree
decision_tree_cm = confusion_matrix(y_test, decision_tree_pred)
decision_tree_disp = ConfusionMatrixDisplay(confusion_matrix=decision_tree_cm, display_labels=["Not Survived", "Survived"])
decision_tree_disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Decision Tree")
plt.savefig("confusion_matrix_decision_tree.png", dpi=300, bbox_inches='tight')
print("Confusion matrix for decision tree saved as: 'confusion_matrix_decision_tree.png'")
plt.show()

# Evaluate and display confusion matrix for Random Forest
random_forest_cm = confusion_matrix(y_test, random_forest_pred)
random_forest_disp = ConfusionMatrixDisplay(confusion_matrix=random_forest_cm, display_labels=["Not Survived", "Survived"])
random_forest_disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")
plt.savefig("confusion_matrix_random_forest.png", dpi=300, bbox_inches='tight')
print("Confusion matrix for random forest saved as: 'confusion_matrix_random_forest.png'")
plt.show()

# ----------------------------------------------------------
# DATAFRAME FOR CONFUSION MATRIX FOR ALL MODELS
# ----------------------------------------------------------

# Ensure that the full DataFrame can be displayed without truncation
pd.set_option('display.max_rows', None)  # Set to None to display all rows
pd.set_option('display.max_columns', None)  # Set to None to display all columns

# Create a DataFrame to display confusion matrix values for each model
confusion_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'True Positive (Survived)': [logreg_cm[1, 1], decision_tree_cm[1, 1], random_forest_cm[1, 1]],
    'False Positive (Not Survived)': [logreg_cm[0, 1], decision_tree_cm[0, 1], random_forest_cm[0, 1]],
    'True Negative (Not Survived)': [logreg_cm[0, 0], decision_tree_cm[0, 0], random_forest_cm[0, 0]],
    'False Negative (Survived)': [logreg_cm[1, 0], decision_tree_cm[1, 0], random_forest_cm[1, 0]],
}

confusion_df = pd.DataFrame(confusion_data)

# Display the DataFrame
print("\n-----------------------------------------\n")
print(confusion_df)
print("\n-----------------------------------------\n")

############################################################
# ----------------------------------------------------------
# IDENTIFY 5 MISCLASSIFIED SAMPLES IN RANDOM FOREST MODEL
# ----------------------------------------------------------
############################################################

# Predict on the test set
y_pred_rf = random_forest_model.predict(X_test)

# Create a DataFrame to store results
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred_rf

# False Positives: predicted 1, actual 0
false_positives = results[(results['Predicted'] == 1) & (results['Actual'] == 0)]

# False Negatives: predicted 0, actual 1
false_negatives = results[(results['Predicted'] == 0) & (results['Actual'] == 1)]

# Reset index for clean display
false_positives = false_positives.reset_index()
false_negatives = false_negatives.reset_index()

# Show first 5 of each
print("\n-----------------------------------------\n")

print("False Positives (Predicted Survived, Actually Did Not):")
print(false_positives.head(5))

print("\n-----------------------------------------\n")

print("\nFalse Negatives (Predicted Not Survived, Actually Did):")
print(false_negatives.head(5))

print("\n-----------------------------------------\n")
