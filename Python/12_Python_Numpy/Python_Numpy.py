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

# Python list of patient ages
patient_ages_list = [29, 35, 46, 52, 38]

# Convert to a NumPy array for efficient computation
patient_ages_array = np.array(patient_ages_list)

print(patient_ages_array)

# Recovery times in days for 5 patients
recovery_times = np.array([5, 7, 3, 4, 6])

print(recovery_times)

# Add a new recovery time
recovery_times = np.append(recovery_times, 8)
print(recovery_times)

# Remove an outlier
recovery_times = np.delete(recovery_times, [2])
print(recovery_times)

# Sort the recovery times
recovery_times = np.sort(recovery_times)

print(recovery_times)

# Simulated gene expression data for 200 samples across 5000 genes
gene_expression_data = np.random.rand(200, 5000)

print(gene_expression_data.shape)

print(gene_expression_data)

# Simulating blood pressure readings for 5 patients over 6 days
bp_readings = np.arange(30)

print("Before reshaping")
print(bp_readings)

bp_readings = bp_readings.reshape(5, 6)

print("After reshaping")
print(bp_readings)

# Extract data for the third patient
third_patient_data = bp_readings[2, :]
print("Third patient:")
print(third_patient_data)

# Compare the first and second blood pressure readings for all patients
first_second_parameters = bp_readings[:, :2]

print("First and second measurements:")
print(first_second_parameters)

# Pre-treatment and post-treatment biomarker levels
pre_treatment = np.array([4.5, 5.0, 5.5])
post_treatment = np.array([5.0, 5.5, 6.0])

increase = post_treatment - pre_treatment

print(increase)

# Heights (in meters) and weights (in kilograms) of 5 patients
heights = np.array([1.75, 1.8, 1.65, 1.9, 1.74])
weights = np.array([70, 80, 60, 90, 75])

bmis = weights / heights**2

print(bmis)

mutations = np.array(['A', 'B', 'A', 'C', 'B', 'A', 'D', 'A'])

unique_mutations = np.unique(mutations)

print(unique_mutations)

mutations = np.array(['A', 'B', 'A', 'C', 'B', 'A', 'D', 'A'])

unique_mutations, counts = np.unique(mutations, return_counts=True)

print(counts)

# Simulating the response of 10 patients to a medication
patient_responses = np.random.randint(0, 11, 10)

print(patient_responses)

# Saving the processed genetic data array
np.save('processed_patient_responses.npy', patient_responses)

# Loading the genetic data array in a new session
loaded_genetic_data = np.load('processed_patient_responses.npy')

# Today's question

myArray = np.ones(15).reshape(3,5)

randomInt = np.random.randint(0, 100, 15)
randomInt = randomInt.reshape(3,5)

myArray = myArray + randomInt

print(myArray)