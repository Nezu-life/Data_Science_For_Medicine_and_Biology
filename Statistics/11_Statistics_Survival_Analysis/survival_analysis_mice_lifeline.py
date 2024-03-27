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
fileName = "../../datasets/survival_vaccinated_mice.csv"
data = pd.read_csv(fileName, sep="\t", header=0)

print(data)
# Kaplan-Meier Estimation
kmf = KaplanMeierFitter()

# Control Group
control_data = data[data['group'] == 'control']
kmf.fit(control_data['lifetimes'], control_data['event_observed'], label='Control Group')
ax = kmf.plot(ci_show=False)

# Vaccinated Group
vaccinated_data = data[data['group'] == 'vaccinated']
kmf.fit(vaccinated_data['lifetimes'], vaccinated_data['event_observed'], label='Vaccinated Group')
kmf.plot(ax=ax, ci_show=False)

# Adding title and labels
plt.title('Survival of Mice: Control vs. Vaccinated Groups')
plt.xlabel('Days Post-Infection')
plt.ylabel('Survival Probability')
plt.ylim(0, 1.05)

# Log-Rank Test
results = logrank_test(control_data['lifetimes'], vaccinated_data['lifetimes'], 
                       event_observed_A=control_data['event_observed'], 
                       event_observed_B=vaccinated_data['event_observed'])

# Plotting the results
plt.show()

# Display Log-Rank Test results
results.print_summary()