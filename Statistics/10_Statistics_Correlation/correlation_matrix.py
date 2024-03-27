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
import sys

# Read the CSV file into a DataFrame, ignoring the first column
fileName = "../../datasets/cancer_cases_worldwide.csv"
data = pd.read_csv(fileName, index_col=0)

# Calculate the correlation matrix of the rows
correlation_matrix = data.T.corr(method="spearman")

# Find the row for Brazil
brazil_correlations = correlation_matrix.loc['Brazil']

# Exclude self-correlation for Brazil
brazil_correlations = brazil_correlations.drop('Brazil', axis=0)

# Find the country with the highest correlation to Brazil
max_cor_country = brazil_correlations.idxmax()
max_cor = brazil_correlations.max()

# Find the country with the lowest correlation to Brazil
min_cor_country = brazil_correlations.idxmin()
min_cor = brazil_correlations.min()

# Print the results
print(f"Highest correlation to Brazil is {max_cor_country}: {max_cor:.2f}.")
print(f"Lowest correlation to Brazil is {min_cor_country}: {min_cor:.2f}.")