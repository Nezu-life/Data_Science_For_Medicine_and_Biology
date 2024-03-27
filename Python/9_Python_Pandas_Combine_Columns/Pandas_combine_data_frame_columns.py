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
fileName = "../../datasets/synthetic_patient_records.csv"
patients = pd.read_csv(fileName)


print(patients.shape)

print(patients.head(5))

print(patients.tail(5))

# Renames one of the columns of the data frame

patients = patients.rename(columns={"Patient ID": "Patient_ID"})

print(patients.head(5))

# Adds a new column (BMI) to the data frame

patients["Height"] = patients["Height"] / 100

patients["BMI"] = patients["Weight"] / patients["Height"] ** 2

print(patients.head(5))

# Removes a column from the data frame

patients = patients.drop("Height", axis=1)

print(patients.head(5))

# Remove all smokers from the data frame and reset indexes

patients = patients[patients['Smoker'] != True]

patients = patients.reset_index(drop=True)

print(patients)

# Lists the number of NaNs (not a numbers) in the data frame

print(patients.isna().sum())

# Removes all rows that have NaNs in the Patient_ID column

patients = patients.dropna(subset=['Patient_ID'])

patients = patients.reset_index(drop=True)

print(patients)

# Reads and concatenates two data frames
URLinputFile_1 = "../../datasets/synthetic_patient_records.csv"
URLinputFile_2 = "../../datasets/synthetic_patient_records_b.csv"

patients_1 = pd.read_csv(URLinputFile_1, sep=",")
patients_2 = pd.read_csv(URLinputFile_2, sep=",")

patients_final = pd.concat([patients_1, patients_2])

patients_final = patients_final.reset_index(drop=True)

print(patients_final.shape)

print(patients_final.head(5))

# Question of lesson
patients = patients.drop("Weight", axis=1)

print(patients.head(5))
