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
from scipy.stats import dunnett
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

# Load the dataset from the command line
fileName = "../../datasets/LDL_cholesterol_synthetic.csv"
ldl = pd.read_csv(fileName)

# Perform one-way ANOVA
F, p = f_oneway(ldl['Control'], ldl['Drug_1'], ldl['Drug_2'], ldl['Drug_3'])

print(f"The ANOVA F-statistic: {F:.2f}")
print(f"The ANOVA p-value: {p:.4f}")

print("")

# The Dunnett's test
res = dunnett(ldl['Drug_1'], ldl['Drug_2'], ldl['Drug_3'], control=ldl['Control'])

print(f"Drug #1 vs control = {res.pvalue[0]:.4f}")
print(f"Drug #2 vs control = {res.pvalue[1]:.4f}")
print(f"Drug #3 vs control = {res.pvalue[2]:.4f}")

# Make a boxplot of the groups

fig, ax = plt.subplots(1, 1)

ax.boxplot([ldl['Control'], ldl['Drug_1'], ldl['Drug_2'], ldl['Drug_3']])
ax.set_xticklabels(["Control", "Drug_1", "Drug_2", "Drug_3"]) 
ax.set_ylabel("LDL Cholesterol levels") 
plt.show()


