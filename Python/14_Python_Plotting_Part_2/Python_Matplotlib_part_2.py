# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all,
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

import matplotlib.pyplot as plt

# Data: Weight and Height of individuals

weight = [60, 65, 70, 75, 80]
height = [160, 165, 170, 175, 180]

plt.scatter(weight, height)

plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Weight vs. Height')

plt.show()

# Plot two sets of data

# Set 1
weight = [60, 65, 70, 75, 80]
height = [160, 165, 170, 175, 180]

# Set 2
weight2 = [55, 60, 65, 70, 75]
height2 = [150, 155, 160, 165, 170]

plt.scatter(weight, height, color='blue', label='Group 1')
plt.scatter(weight2, height2, color='red', label='Group 2')

plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Weight vs. Height by Group')

plt.legend()

plt.show()

# Using colorsmaps to color a scatterplot

weight = [60, 75, 80, 55, 63]
height = [160, 165, 170, 160, 170]

age = [15, 30, 27, 70, 55]

# Using the same data from the previous scatter plot example
plt.scatter(weight, height, c=age, cmap='viridis')

plt.colorbar()

plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Weight vs. Height Colored by Weight')

plt.show()

# Set size of dots according to age

weight = [60, 75, 80, 55, 63]
height = [160, 165, 170, 160, 170]
age = [15, 30, 27, 70, 55]

plt.scatter(weight, height, s=age)

plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Weight vs. Height with Larger Dots')

plt.show()

# Barplot of heart rates

drugs = ["Drug A", "Drug B", "Drug C"]
heart_rates = [60, 100, 150]

plt.bar(drugs, heart_rates)

plt.ylabel('Average Heart Rate (bpm)')
plt.title('Heart Rate After Treatment')

plt.show()

# Horizontal barplot of heart rates

drugs = ["Drug A", "Drug B", "Drug C"]
heart_rates = [60, 100, 150]

plt.barh(drugs, heart_rates)

plt.xlabel('Average Heart Rate (bpm)')
plt.title('Heart Rate After Treatment')

plt.show()

# Piechart of time spent on each activities

labels = ['Resting', 'Walking', 'Running']
sizes = [5, 3, 2]

plt.pie(sizes, labels=labels, autopct='%1.1f%%')

plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular

plt.show()

# The "perfect plot"

heart_rates = [60, 100, 150]
blood_pressure = [80, 120, 140]

plt.plot(heart_rates, label='Heart Rate', marker="o")
plt.plot(blood_pressure, label='Blood Pressure', marker="o")

plt.legend()

plt.grid(True)

plt.xlabel('Activity')

plt.ylabel('Measurement')
plt.title('Physiological Measurements During Different Activities')

plt.show()

heart_rates = [60, 100, 150]
blood_pressure = [80, 120, 140]

plt.plot(heart_rates, label='Heart Rate', marker="o")
plt.plot(blood_pressure, label='Blood Pressure', marker="o")

plt.grid(True)

plt.show()