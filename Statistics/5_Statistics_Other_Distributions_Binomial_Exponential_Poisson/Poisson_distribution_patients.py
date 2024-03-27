# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

from scipy.stats import poisson
import matplotlib.pyplot as plt

# Average number of patients arriving per hour (lambda)
lambda_patients = 5

# Maximum number of patients we want to plot
max_patients = 15

# Calculate the probability of seeing 0, 1, 2, ..., max_patients patients in an hour
patients = range(0, max_patients + 1)
probabilities = [poisson.pmf(k, lambda_patients) for k in patients]

# Create a bar plot to visualize this
plt.bar(patients, probabilities)
plt.title('Probability of Patient Arrivals per Hour in a Hospital ED')
plt.xlabel('Number of Patients')
plt.ylabel('Probability')
plt.xticks(patients)
plt.show()

