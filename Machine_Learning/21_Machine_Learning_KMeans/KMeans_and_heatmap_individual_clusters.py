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
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def main(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Separate the 'Country' column
    countries = df.iloc[:, 0]
    df_data = df.drop(df.columns[0], axis=1)
    
    # Fill missing values with the mean of the columns
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_data), columns=df_data.columns)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=15, random_state=42)
    clusters = kmeans.fit_predict(df_imputed)
    
    # Add the cluster labels and country names back to the DataFrame
    df_imputed['Cluster'] = clusters
    df_imputed['Country'] = countries
    
    # Determine the global min and max values across the entire DataFrame (excluding 'Country' and 'Cluster')
    vmin = df_imputed.drop(['Country', 'Cluster'], axis=1).values.min()
    vmax = df_imputed.drop(['Country', 'Cluster'], axis=1).values.max()

    # Plot and save a heatmap for each cluster using the same color scale
    for cluster_num in range(15):
        # Filter the DataFrame for the current cluster
        cluster_data = df_imputed[df_imputed['Cluster'] == cluster_num]
        
        # Sort the data by country name for better visualization
        cluster_data.sort_values('Country', inplace=True)
        
        # Extract and sort the country names for labeling
        country_labels = cluster_data['Country'].values
        
        # Drop the 'Country' and 'Cluster' columns to only visualize the data in the heatmap
        cluster_data = cluster_data.drop(['Country', 'Cluster'], axis=1)
        
        # Create a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cluster_data, cmap='Reds', annot=False, cbar_kws={'label': 'Scale'}, yticklabels=country_labels, vmin=vmin, vmax=vmax)
        plt.title(f'Heatmap of Cluster {cluster_num}')
        plt.yticks(rotation=0)  # Keep the country labels horizontal for readability
        
        # Save the heatmap as a PNG file
        plt.savefig(f'cluster_{cluster_num}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to avoid displaying it in the notebook

    print("Heatmaps saved as PNG files.")

if __name__ == "__main__":
    inputFile = "../../datasets/child_mortality.csv"
    main(inputFile)
