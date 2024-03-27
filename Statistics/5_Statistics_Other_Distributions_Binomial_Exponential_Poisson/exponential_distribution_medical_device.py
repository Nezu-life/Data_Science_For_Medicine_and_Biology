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

lambda_param = 1 / 15  # Life-span of pacemaker
time_years = 5  # Time to check the lifespan against

# Calculate the probability
probability = np.exp(-lambda_param * time_years)

print(f"Probability that a pacemaker lasts {time_years} years: {probability:.4f}")


