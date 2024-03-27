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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset from sklearn
data = load_breast_cancer()
X = data.data
y = data.target

# Preprocess the data
X = StandardScaler().fit_transform(X)

# Linear Regression
linear_regressor = LinearRegression().fit(X, y)
y_pred_linear = linear_regressor.predict(X)

# Logistic Regression
log_regressor = LogisticRegression().fit(X, y)
y_pred_log = log_regressor.predict_proba(X)[:, 1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Replace numeric target values with 'M' and 'B' for the legend
label_map = {0: 'B', 1: 'M'}
labels = np.vectorize(label_map.get)(y)

# Linear Regression Plot
scatter = ax1.scatter(X[:, 0], y_pred_linear, c=y, cmap='winter')
legend1 = ax1.legend(handles=scatter.legend_elements()[0], labels=['B', 'M'], title="Classes")
ax1.add_artist(legend1)
ax1.set_title("Linear Regression Predictions")

# Logistic Regression Plot
scatter = ax2.scatter(X[:, 0], y_pred_log, c=y, cmap='autumn')
legend2 = ax2.legend(handles=scatter.legend_elements()[0], labels=['B', 'M'], title="Classes")
ax2.add_artist(legend2)
ax2.set_title("Logistic Regression Predictions")

plt.tight_layout()
plt.show()
