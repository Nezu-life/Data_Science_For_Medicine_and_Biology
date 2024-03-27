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

# Group by Gender, then get the max weight values

print(patients.groupby(["Gender"]).Weight.max())

# Group by Blood_Type, then get the mean weight values

print(patients.groupby(["Blood_Type"]).Weight.mean())

# Group by Blood_Type and Gender, then get the mean weight values

print(patients.groupby(["Gender", "Blood_Type"]).Weight.mean())

# Sort dataset by the Age feature

print(patients.sort_values(by="Age"))

# Sort the dataset by Age and Cholesterol

print(patients.sort_values(by=["Age", "Cholesterol"]))

# Sort the dataset by Age and Cholesterol. Decreasing order.

print(patients.sort_values(by=["Age", "Cholesterol"], ascending=False))

# Today's question

sorted_df = patients.groupby("Blood_Type").Weight.mean()

print(sorted_df)