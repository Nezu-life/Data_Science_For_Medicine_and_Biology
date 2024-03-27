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
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Read the CSV file
csv_file = "../../datasets/maternal_mortality_south_america.csv"
data = pd.read_csv(csv_file)

# Use the first column as labels, and drop it from the data used for clustering
labels = data.iloc[:, 0].values
data = data.drop(data.columns[0], axis=1)

# Perform hierarchical/agglomerative clustering
Z = linkage(data, method='average')

# Plot the dendrogram using the labels from the first column
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=labels, orientation="left")
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Instance')
plt.ylabel('Distance')
plt.show()

