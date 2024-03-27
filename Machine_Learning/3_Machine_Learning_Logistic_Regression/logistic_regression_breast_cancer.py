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
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data[:, 0].reshape(-1, 1)  # using only the first feature for visualization
y = data.target

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# Predict probabilities
x_vals = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 300)
y_probs = clf.predict_proba(x_vals.reshape(-1, 1))[:, 1]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_train_scaled[y_train == 0], y_train[y_train == 0], color='blue', label='B')
plt.scatter(X_train_scaled[y_train == 1], y_train[y_train == 1], color='red', label='M')
plt.plot(x_vals, y_probs, color='black')
plt.xlabel('Feature Value (scaled)')
plt.ylabel('Class label')
plt.title('Logistic Regression Fit')
plt.legend()
plt.show()
