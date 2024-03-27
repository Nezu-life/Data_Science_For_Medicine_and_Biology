# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

from scipy.stats import binom

# Number of trials (people vaccinated), 
# Number of successes (effective vaccinations)
# Probability of success
n, k, p = 10, 6, 0.7

# Probability of exactly 6 out of 10 successful vaccinations
probability = binom.pmf(k, n, p)
print("Probability of exactly 6 successful vaccinations:", probability)


