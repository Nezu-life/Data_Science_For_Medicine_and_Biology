# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

#####################################################
### WATCH OUT!
### Maybe you have to install the lifelines package
#####################################################

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Reads the input file (CSV)
fileName = "../../datasets/survival_lung_cancer.csv"
data = pd.read_csv(fileName, sep=",", header=0)

# Kaplan-Meier Estimation
kmf = KaplanMeierFitter()

# Non-Small Cell Lung Cancer
nsclc_data = data[data['group'] == 'NSCLC']
kmf.fit(nsclc_data['lifetimes'], nsclc_data['event_observed'], label='Non-Small Cell Lung Cancer')
ax = kmf.plot(ci_show=False)

# Small Cell Lung Cancer
sclc_data = data[data['group'] == 'SCLC']
kmf.fit(sclc_data['lifetimes'], sclc_data['event_observed'], label='Small Cell Lung Cancer')
kmf.plot(ax=ax, ci_show=False)

# Adding title and labels
plt.title('Survival of Lung Cancer Patients')
plt.xlabel('Days')
plt.ylabel('Survival Probability')
plt.ylim(0, 1.05)

# Log-Rank Test
results = logrank_test(nsclc_data['lifetimes'], sclc_data['lifetimes'], 
                       event_observed_A=nsclc_data['event_observed'], 
                       event_observed_B=sclc_data['event_observed'])

# Plotting the results
plt.show()

# Display Log-Rank Test results
results.print_summary()
