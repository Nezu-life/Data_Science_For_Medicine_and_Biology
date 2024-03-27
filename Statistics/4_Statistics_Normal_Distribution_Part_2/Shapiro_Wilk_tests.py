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

# Generate 1000 points normally distributed
normal_data = np.random.normal(0, 1, 100)

# Generate 1000 points non-normally distributed (using a uniform distribution)
non_normal_data = np.random.uniform(-3, 3, 100)

# Calculating Shapiro-Wilk test for both datasets
w_normal, p_normal = stats.shapiro(normal_data)
w_non_normal, p_non_normal = stats.shapiro(non_normal_data)

print("Normally Distributed Data: W =", w_normal, ", p-value =", p_normal)
print("Non-Normal Data: W =", w_non_normal, ", p-value =", p_non_normal)

# Creating a figure with two panels for the histograms
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Histogram for normally distributed data
axs[0].hist(normal_data, bins=30, alpha=0.7, color='blue', label='Normal')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')
axs[0].legend()

# Histogram for non-normally distributed data
axs[1].hist(non_normal_data, bins=30, alpha=0.7, color='red', label='Non-Normal')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')
axs[1].legend()

# Adjusting layout
plt.tight_layout()
plt.show()

