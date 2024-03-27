# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

# Import necessary libraries
import pandas as pd  # Pandas is used for data manipulation and analysis
import matplotlib.pyplot as plt  # Matplotlib's pyplot is used for creating plots and charts

# Load the CSV file into a DataFrame
filename = '../../datasets/breast_cancer_diagnostic.csv'
data = pd.read_csv(filename)  # Read the CSV file

# Extract the 'radius_mean' column from the DataFrame
radius_mean = data['radius_mean']  # Extracting the column by its name

# Create a histogram for 'radius_mean'
# A histogram is a representation of the distribution of a continuous data set.
plt.figure(figsize=(10,6))  # Set the size of the figure
plt.hist(radius_mean, bins=30, color='skyblue', edgecolor='black')  # Set the number of bins, color, and edgecolor
plt.title('Histogram of Radius Mean')  # Set the title of the histogram
plt.xlabel('Radius Mean')  # Set the label for the x-axis
plt.ylabel('Frequency')  # Set the label for the y-axis

# Display the histogram
plt.show()  # This command will display the histogram
