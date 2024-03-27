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

# Selecting only one feature of our dataset

bloodTypes = patients["Blood_Type"]

bloodTypes = patients.Blood_Type # Same as above

print(bloodTypes)


# Prints only the first row of the patients data frame
thisPatient = patients.iloc[0]

print(thisPatient)


# Prints only the 10th row of the patients data frame
thisPatient = patients.iloc[9]

print(thisPatient)


# Prints only the rows 3-5 of the patients data frame
thisPatient = patients.iloc[0:5]

print(thisPatient)


# Prints rows 0-10 of the patients data frame
thisPatient = patients.iloc[0:9]

print(thisPatient)

# Prints rows 0-5 of the Height column of the patients data frame
thisPatient = patients.iloc[:5, 3]

print(thisPatient)


# Selects only some columns and rows from a data frame
miniDF = patients.loc[:100, ["Patient ID", "Age", "Height", "Weight"]]

print(miniDF.head(5))


# Creates a data frame with older female patients

olderFemales = patients.loc[(patients["Age"] > 80) & (patients["Gender"] == "Female")]

print(olderFemales)


# Selects patients who have O+ or O- blood types

o_type_only = patients.loc[patients["Blood_Type"].isin(["O+", "O-"])]

print(o_type_only)
