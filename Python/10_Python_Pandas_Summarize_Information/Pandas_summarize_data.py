# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all,
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

# Import a large dataset from the Nezu Life Sciences repository
# Print its "shape" (number of rows and columns)
# and its first 5 rows.

import pandas as pd

# Reads an input file
fileName = "../../datasets/synthetic_patient_records.csv"
patients = pd.read_csv(fileName)

print(patients.head(5))

# Summarize the info of the Age column
print(patients.describe())

# Prints the average of the cholesterol values

mean_cholesterol = patients["Cholesterol"].mean()

print(mean_cholesterol)

# Compares the average cholesterol of male and female patients

male_chol = patients.loc[patients["Gender"] == "Male"]["Cholesterol"].mean()

female_chol = patients.loc[patients["Gender"] == "Female"]["Cholesterol"].mean()

print("Male: ", male_chol, "Female:", female_chol)

# Print unique blood types

blood_types = patients['Blood_Type'].unique()

print(blood_types)

# How often did each blood type occur in the dataset?

print(patients["Blood_Type"].value_counts())

# A question for this lesson

print(patients['Diabetes'].value_counts())