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
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
filename = "../../datasets/two_groups_unpaired_t-test.csv"
data = pd.read_csv(filename)


# Read CSV file
data = pd.read_csv(filename)

# Perform unpaired two-sided t-test
t_statistic, p_value = stats.ttest_ind(data['Placebo'], data['Drug'], equal_var=True)

print(f"T-Statistic: {t_statistic}, P-Value: {p_value}")

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, palette="Set2")
plt.title("Bodyweight comparison")
plt.ylabel("Bodyweight [kg]")
plt.grid(True)
plt.show()

