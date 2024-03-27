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
from scipy.stats import chi2_contingency
import sys

# Load the dataset
fileName = "../../datasets/lab_animal_diet_preference.csv"
data = pd.read_csv(fileName, index_col=0)
print(data)

chi2, p, dof, expected = chi2_contingency(data)

# Print the results
print("\nResults of the Chi-Squared Test:")
print(f"Chi-Squared Statistic: {chi2:.2f}")
print(f"P-value: {p:.4f}")
