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

# Simple line plot
heart_rates = [60, 100, 150]

plt.plot(heart_rates)

plt.show()

# Simple line plot with a marker

heart_rates = [60, 100, 150]

plt.plot(heart_rates, marker='o')

plt.show()

# Change the color to red

heart_rates = [60, 100, 150]

plt.plot(heart_rates, marker='o', color='red')

plt.show()

# Change the line style
heart_rates = [60, 100, 150]

plt.plot(heart_rates, marker='o', color='red', linestyle='--')

plt.show()

# Change the line width

heart_rates = [60, 100, 150]

plt.plot(heart_rates, marker='o', color='red', linewidth=3)

plt.show()

# Plot more than one set of data

blood_pressure = [80, 120, 140]
heart_rates = [60, 100, 150]

plt.plot(heart_rates, label='Heart Rate')
plt.plot(blood_pressure, label='Blood Pressure')

plt.show()

# Add a grid to the plot
blood_pressure = [80, 120, 140]
heart_rates = [60, 100, 150]

plt.plot(heart_rates, label='Heart Rate')
plt.plot(blood_pressure, label='Blood Pressure')

plt.grid(True)

plt.show()

# Add a vertical grid to the plot

blood_pressure = [80, 120, 140]
heart_rates = [60, 100, 150]

plt.plot(heart_rates, label='Heart Rate')
plt.plot(blood_pressure, label='Blood Pressure')

plt.grid(True, axis='x')

plt.show()

# Additional data: Average blood pressure during different activities
blood_pressure = [80, 120, 140]
heart_rates = [60, 100, 150]

plt.plot(heart_rates, label='Heart Rate')
plt.plot(blood_pressure, label='Blood Pressure')

plt.grid(True, axis='y')

plt.show()

# Add labels and a title to the plot
heart_rates = [60, 100, 150]

plt.plot(heart_rates, label='Heart Rate')

plt.xlabel('Activity')
plt.ylabel('Average Heart Rate (bpm)')
plt.title('Heart Rate During Different Activities')

plt.show()

# Subplots, side-by-side

blood_pressure = [80, 120, 140]
heart_rates = [60, 100, 150]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(heart_rates)
axs[0].set_title('Heart Rate')

axs[1].plot(blood_pressure)
axs[1].set_title('Blood Pressure')

plt.show()

# Subplots, on top of each other

blood_pressure = [80, 120, 140]
heart_rates = [60, 100, 150]

fig, axs = plt.subplots(2, 1, figsize=(5, 10))

axs[0].plot(heart_rates)
axs[0].set_title('Heart Rate')

axs[1].plot(blood_pressure)
axs[1].set_title('Blood Pressure')

plt.show()