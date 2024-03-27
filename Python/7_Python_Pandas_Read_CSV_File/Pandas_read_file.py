# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

import pandas as pd

# Reads an input file
# Reads an input file
fileName = "../../datasets/synthetic_patient_records.csv"
patients = pd.read_csv(fileName, sep="\t")


print(patients.shape)

print(patients.head(5))

print(patients.tail(5))

print(patients.dtypes)
print(patients.info())