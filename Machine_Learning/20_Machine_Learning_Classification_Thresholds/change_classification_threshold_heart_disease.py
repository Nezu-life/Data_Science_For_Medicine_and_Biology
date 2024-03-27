# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
fileName = "../../datasets/heart_disease.csv"
data = pd.read_csv(fileName)

# Separate features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:,-1:].values

# Initialize StratifiedKFold for 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Lists to store real values and predicted probabilities for all folds
all_real_values = []
all_predicted_probs = []

# Perform 10-fold stratified cross-validation
for train_index, test_index in skf.split(X, y):
    # Split data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the Decision Tree classifier
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)

    # Predict probabilities on the test set
    y_probs = clf.predict_proba(X_test)[:, 1]

    # Store real values and predicted probabilities
    all_real_values.extend(y_test)
    all_predicted_probs.extend(y_probs)

# Convert lists to numpy arrays for metric calculation
all_real_values = np.array(all_real_values)
all_predicted_probs = np.array(all_predicted_probs)

# Calculate metrics for different thresholds
thresholds = np.arange(0.1, 0.90, 0.01)
balanced_acc_scores = []
f1_scores = []

# Changes the threshold and calculates classification measures
for threshold in thresholds:
    # Apply threshold to probabilities
    thresholded_predictions = (all_predicted_probs >= threshold).astype(int)

    # Calculate metrics
    balanced_acc = balanced_accuracy_score(all_real_values, thresholded_predictions)
    f1 = f1_score(all_real_values, thresholded_predictions)

    balanced_acc_scores.append(balanced_acc)
    f1_scores.append(f1)

# Time to plot everything
plt.figure(figsize=(10, 12))

# Balanced accuracy plot
plt.subplot(2, 1, 1)  # Adjusted to be 2 rows, 1 column, 1st plot
plt.plot(thresholds, balanced_acc_scores, marker='o')
plt.title('Balanced Accuracy at Different Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Balanced Accuracy')

# F1
plt.subplot(2, 1, 2)  # Adjusted to be 2 rows, 1 column, 2nd plot
plt.plot(thresholds, f1_scores, marker='o', color='orange')
plt.title('F1 Score at Different Thresholds')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')

plt.tight_layout()
plt.show()