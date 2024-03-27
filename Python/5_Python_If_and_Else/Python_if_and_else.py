# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

condition = True

if(condition):
  # do something
  print("The condition is true.")
else:
  # do something else
  print("The condition is false.")

hemoglobin_level = 10 # measured in g/dL

if(hemoglobin_level < 12):
    print("The patient may have anemia.")
else:
    print("The patient's hemoglobin level is normal.")

colonies_count = 150

if(colonies_count > 100):
    print("High bacterial growth.")
else:
    print("Normal bacterial growth.")

patient_age = 45

if(patient_age < 12):
  print("10mg")  # Pediatric dosage
elif(patient_age < 18):
  print("20mg")  # Adolescent dosage
elif(patient_age < 65):
  print("30mg")  # Adult dosage
else:
  print("25mg")  # Senior dosage

# Example with logical operator 'and'

glucose_level = 110  # mg/dL
has_symptoms = True

if(glucose_level > 100 and has_symptoms):
    print("High risk of diabetes.")
else:
    print("No diabetes risk at the moment.")

# Example with logical operator 'or'

patient_coughing = False
patient_fever = True

if(patient_coughing or patient_fever):
    print("Patient requires medical evaluation for possible infection.")
else:
    print("Symptoms do not indicate immediate concern.")

# Bacterial infection (and not example)
    
bacteria_type = "E. coli"
resistant_to_amoxicillin = False

if(bacteria_type == "E. coli" and not resistant_to_amoxicillin):
    print("Recommend amoxicillin treatment.")
else:
    print("Further antibiotic susceptibility testing needed.")

# Example with and
    
patient_age = 68
patient_overweight = True

if(patient_age > 65 and patient_overweight == True):
  print("The senior patient is overweight.")
elif(patient_age <= 65 and patient_overweight == True):
  print("The patient is overweight.")
else:
  print("The patient is not senior and not overweight.")
