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
import pandas as pd  
import matplotlib.pyplot as plt  

# Specify the name of the CSV file containing the data
filename = '../../datasets/breast_cancer_diagnostic.csv'

# Read the CSV file using pandas and store the data in a DataFrame
df = pd.read_csv(filename)  

# Specify the names of the columns for which we will create histograms
columns = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
    "smoothness_mean", "compactness_mean", "concavity_mean"
]

# Specify different pastel colors for each histogram
colors = ['lightcoral', 'peachpuff', 'lavender', 'palegreen', 'paleturquoise', 'moccasin', 'lightpink']

# Create a figure with 4 rows and 2 columns of subplots, and set the figure size
fig, axs = plt.subplots(4, 2, figsize=(10, 20))  

# Flatten the 2D array of axes into a 1D array, so we can easily loop over it
axs = axs.flatten()  

# Loop over each column name, color, and subplot axis to create histograms
for col, color, ax in zip(columns, colors, axs):
    ax.hist(df[col], color=color, bins=30, edgecolor='k', alpha=0.7)  
    
    # Set the title, x-axis label, and y-axis label of each subplot
    ax.set_title(col)  
    ax.set_xlabel('')  # No label on x-axis
    ax.set_ylabel('Frequency')  

# Turn off the last subplot as it will remain empty (we have 7 histograms but 8 subplots)
axs[-1].axis('off')  

# Adjust the space between the histograms
plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjusted space between subplots

# Adjust the layout of the subplots in the figure to ensure that they fit well
plt.tight_layout()

# Display the complete figure containing all histograms
plt.show()  
