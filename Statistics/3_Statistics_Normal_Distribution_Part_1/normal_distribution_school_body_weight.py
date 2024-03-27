# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate a random sample of 1000 values with a normal distribution
population = np.random.normal(loc=60, scale=10, size=1000)
population_mean = np.mean(population)

# Select a random sample of 50 from the population
sample = np.random.choice(population, size=50, replace=False)

# Calculate the mean of the sample
sample_mean = np.mean(sample)

# Calculate the standard error of the sample
standard_error = np.std(sample, ddof=1) / np.sqrt(len(sample))

# Calculate the 95% confidence interval
confidence_interval = stats.norm.interval(0.95, loc=sample_mean, scale=standard_error)

# Create a histogram comparing the distribution of the sample of 1000 to the sample of 50
plt.figure(figsize=(10, 6))

plt.hist(population, bins=30, alpha=0.5, label='Population (1000 values)')
plt.hist(sample, bins=30, alpha=0.5, label='Sample (50 values)')

plt.axvline(population_mean, color='blue', linestyle='dashed', linewidth=2, label='Population Mean')
plt.axvline(sample_mean, color='orange', linestyle='dashed', linewidth=2, label='Sample Mean')

plt.title('Comparison of Population and Sample Distributions')

plt.xlabel('Values')
plt.ylabel('Frequency')

plt.legend()
plt.show()

print(f"Population mean: {population_mean}")
print(f"Sample mean: {sample_mean}")
print(f"Sample SE: {standard_error}")
print(f"95% Confidence Interval: {confidence_interval}")

