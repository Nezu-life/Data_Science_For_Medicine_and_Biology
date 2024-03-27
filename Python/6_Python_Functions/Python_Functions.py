# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

def greet():
    print("Hello, welcome to Python functions!")


greet()  # Calls the greet function

def greet_name(name):
    print("Hello,", name, "welcome to Python functions!")

greet_name("Amie")


def add(a, b):
    return a + b

result = add(5, 3)

print(result)

def greet_someone(name="User"):
    print("Hello,", name, "welcome to Python functions!")

greet_someone()

greet_someone("Linus")

def full_name(first, last):
    return(first, last)

name = full_name(last="Doe", first="John")

print(name)

def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)

    return bmi

def pounds_to_kilograms(pounds):
    kilograms = pounds * 0.453592

    return kilograms

athlete_weight = 250 # Weight in pounds

athlete_weight_kg = pounds_to_kilograms(athlete_weight)

bmi = calculate_bmi(athlete_weight_kg, 1.8)

print(bmi)

def question_function(x, y, z):
  tmp = x + y

  result = tmp * z

  return(result)

super_number = question_function(2, 3, 5)

print(super_number)

