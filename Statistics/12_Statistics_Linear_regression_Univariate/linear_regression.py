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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CSV file into a DataFrame
fileName = "../../datasets/HIV_world.csv"
df = pd.read_csv(fileName)

# Reshape the 'year' column to use it in sklearn
X = df['year'].values.reshape(-1, 1)
y = df['cases'].values

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the original data and the regression line
plt.scatter(X, y, color='darkblue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')

plt.xlabel('Year')
plt.ylabel('Cases')
plt.title('Linear Regression on HIV Incidence Data')
plt.legend()

plt.show()  # Uncomment this line if you want to display the plot when running the script

# Print the regression equation and parameters
equation = f"cases = {model.intercept_:.2f} + {model.coef_[0]:.2f}*year"
print(equation)

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient for year: {model.coef_[0]}")
