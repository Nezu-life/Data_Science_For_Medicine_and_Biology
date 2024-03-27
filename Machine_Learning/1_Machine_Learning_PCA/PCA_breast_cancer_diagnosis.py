# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("../../datasets/breast_cancer_diagnostic_PCA.csv")

# Ignore the first column
X = data.iloc[:, 1:-1]
y = data['diagnosis']

# Color map for the diagnosis
color_map = {'M': 'red', 'B': 'blue'}
colors = y.map(color_map)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(211, projection='3d')  # Change from 121 to 211
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=colors)
ax1.set_title("Original 3D Data")

# Perform PCA and transform data to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a 2D scatter plot
ax2 = fig.add_subplot(212)  # Change from 122 to 212
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)
ax2.set_title("PCA 2D Projection")

plt.tight_layout()  # This ensures that the plots are spaced appropriately
plt.show()
