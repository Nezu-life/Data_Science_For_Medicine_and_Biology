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
import pandas as pd  # pandas is a library for handling tabular data
import matplotlib.pyplot as plt  # matplotlib.pyplot is used for creating plots and charts

# Load the CSV file
filename = '../../datasets/breast_cancer_diagnostic.csv'  # Assigning the name of the file to a variable called 'filename'
df = pd.read_csv(filename)  # Using pandas to read the CSV file and store it in a variable called 'df'

# Define a figure with 2 subplots, arranged vertically
fig, axs = plt.subplots(2, 1, figsize=(7, 10))  # Creating a figure with 2 subplots (2 rows, 1 column) and specifying the size

# Set color for each histogram
colors = ['peachpuff', 'lightblue']  # Assigning colors for the histograms in a list called 'colors'

# Set title for each histogram
titles = ['Diagnosis: M', 'Diagnosis: B']  # Assigning titles for the histograms in a list called 'titles'

# Calculate common bins for both histograms
min_val = df['radius_mean'].min()  # Finding the minimum value in the 'radius_mean' column of the dataframe 'df'
max_val = df['radius_mean'].max()  # Finding the maximum value in the 'radius_mean' column of the dataframe 'df'
bins = plt.np.linspace(min_val, max_val, 30)  # Creating 30 equally spaced bins between the min and max values for the histograms

# Loop over the subplots, colors, and titles to create histograms
for ax, color, title in zip(axs, colors, titles):  # Iterating over each subplot, color, and title
    # Filter the dataframe by the diagnosis and plot histogram of 'radius_mean'
    ax.hist(df[df['diagnosis'] == title[-1]]['radius_mean'], bins=bins, color=color, edgecolor='k', alpha=0.7)  # Plotting the histogram with the specified color, edge color, and transparency level
    ax.set_title(title)  # Setting the title for each subplot
    ax.set_xlabel('Radius Mean')  # Labeling the x-axis as 'Radius Mean'
    ax.set_ylabel('Frequency')  # Labeling the y-axis as 'Frequency'

# Adjust the layout and display the plot
plt.tight_layout()  # Adjusting the layout so everything fits nicely
plt.show()  # Displaying the plots
