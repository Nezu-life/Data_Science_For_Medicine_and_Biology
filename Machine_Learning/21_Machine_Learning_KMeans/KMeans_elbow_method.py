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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def main(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Drop the 'Country' column or the first column
    df = df.drop(df.columns[0], axis=1)

    # Fill missing values with the mean of the columns
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Determine the range of cluster counts to evaluate
    cluster_range = range(5, 50)

    # Apply the elbow method
    inertia_values = []
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_filled)
        inertia_values.append(kmeans.inertia_)

    # Plot the elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertia_values, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.xticks(cluster_range)
    plt.show()

if __name__ == "__main__":
    inputFile = "../../datasets/child_mortality.csv"
    main(inputFile)
