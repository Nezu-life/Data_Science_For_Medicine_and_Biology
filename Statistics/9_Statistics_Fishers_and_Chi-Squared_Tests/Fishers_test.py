# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

import sys
import pandas as pd
from scipy.stats import fisher_exact

# Load the dataset
fileName = "../../datasets/side_effect_ears_bigger.csv"
data = pd.read_csv(fileName)

# Assuming the data has two categorical columns
# The contingency table is formed by cross-tabulating these columns
contingency_table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])

# Printing the contingency table
print("Contingency Table:")
print(contingency_table)

# Performing Fisher's Exact Test
odds_ratio, p_value = fisher_exact(contingency_table)

# Printing the results
print("\nResults of Fisher's Exact Test:")
print(f"Odds Ratio: {odds_ratio}")
print(f"P-Value: {p_value}")


