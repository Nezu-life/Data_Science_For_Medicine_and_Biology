# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024


###############################################################
### WATCH OUT!
### Maybe you need to install the statsmodels package
###############################################################

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_data.csv' with the path to your CSV file
fileName = "../../datasets/systolic_blood_pressure.csv"
data = pd.read_csv(fileName)

# Two-way ANOVA
model = ols('Systolic ~ C(Sex) + C(Treatment) + C(Sex):C(Treatment)', data=data).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print("Two-Way ANOVA result:\n", anova_result)

# Tukey's HSD test for pairwise comparisons
tukey = pairwise_tukeyhsd(endog=data['Systolic'], groups=data['Sex'] + data['Treatment'], alpha=0.05)
print("\nTukey's Test Result:\n", tukey)

# Save Tukey's test result to CSV
tukey_result_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
tukey_result_df.to_csv("tukey_test_results.csv", index=False)

## Create a grouped boxplot
ax = sns.boxplot(x='Treatment', y='Systolic', hue='Sex', data=data, palette='Set2')
ax.set(xlabel='', ylabel='Systolic Blood Pressure [mm Hg]')
plt.show()