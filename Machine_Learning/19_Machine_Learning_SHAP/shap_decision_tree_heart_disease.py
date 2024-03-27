# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

import shap
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import sys

# Load the heart disease dataset
fileName = "../../datasets/heart_disease.csv"
data = pd.read_csv(fileName)

X = data.drop(['disease'], axis=1)
y = data['disease']

# Train an XGBoost model on the entire dataset
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X, y)

# Explain the model's predictions using SHAP
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Plot the SHAP bar plot with clustering
plt.figure(figsize=(10, 4))
shap.plots.bar(shap_values, max_display=10)

# Plot the SHAP beeswarm plot
plt.figure(figsize=(10, 4))

shap.plots.beeswarm(shap_values, max_display=10)


plt.show()
plt.tight_layout()