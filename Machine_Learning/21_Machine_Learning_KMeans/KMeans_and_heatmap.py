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
import sys
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Read the CSV file
inputFile = "../../datasets/child_mortality.csv"
df = pd.read_csv(inputFile)

# Separate the 'Country' column
countries = df.iloc[:, 0]
df_data = df.drop(df.columns[0], axis=1)

# Fill missing values with the mean of the columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_data), columns=df_data.columns)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=15, random_state=42)
df_imputed['cluster'] = kmeans.fit_predict(df_imputed)

# Add the country labels to the DataFrame for later use in plotting
df_imputed['Country'] = countries

# Sort the DataFrame by the cluster labels
df_imputed.sort_values('cluster', inplace=True)
sorted_countries = df_imputed['Country']

# Remove the 'Country' and 'cluster' columns to only visualize the data in the heatmap
df_imputed.drop(['Country', 'cluster'], axis=1, inplace=True)

# Set the font size for the row labels (countries)
sns.set(rc={'ytick.labelsize': 2})

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_imputed, cmap='Reds', annot=False, cbar_kws={'label': 'Scale'}, yticklabels=sorted_countries)
plt.title('Heatmap of Columns Organized by KMeans Cluster')
plt.yticks(rotation=0)  # Keep the country labels horizontal for readability
plt.show()

