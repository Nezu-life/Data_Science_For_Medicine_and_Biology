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
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from scipy.stats import tukey_hsd

# Load the dataset from the command line
fileName = "../../datasets/LDL_cholesterol_synthetic.csv"
ldl = pd.read_csv(fileName)

# Perform one-way ANOVA
F, p = f_oneway(ldl['Control'], ldl['Drug_1'], ldl['Drug_2'], ldl['Drug_3'])

print(f"The ANOVA F-statistic: {F:.2f}")
print(f"The ANOVA p-value: {p:.4f}")

print("")

# The Dunnett's test
res = tukey_hsd(ldl['Control'], ldl['Drug_1'], ldl['Drug_2'], ldl['Drug_3'])

print(res)

# Make a boxplot of the groups

fig, ax = plt.subplots(1, 1)

ax.boxplot([ldl['Control'], ldl['Drug_1'], ldl['Drug_2'], ldl['Drug_3']])
ax.set_xticklabels(["Control", "Drug_1", "Drug_2", "Drug_3"]) 
ax.set_ylabel("LDL Cholesterol levels") 
plt.show()


