# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Assuming 'heart_disease.csv' is in your working directory and properly formatted
# Load the heart disease dataset
data = pd.read_csv('../../datasets/heart_disease.csv')
X = data.drop("disease", axis=1)
y = data["disease"]

# Initialize 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []

for train_index, test_index in kf.split(X, y):
    # Splitting data into training and test sets for the current fold using .iloc for row selection
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Standardizing the features based on the training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predicting and evaluating the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calculating average accuracy

average_accuracy = np.mean(accuracies)

print(f"Average accuracy:", average_accuracy)
