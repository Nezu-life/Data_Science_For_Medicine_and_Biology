# Data Science for Medicine and Biology
#
# Nezu Life Sciences
#
# Feel free to modify, redistribute and above all, 
# create something with this code.
#
# Tiago Lopes, PhD
# March 2024

########################################################
### WATCH OUT!
### Maybe you need to install the pmdarima package
########################################################

from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import numpy as np

# Reads the input dataset
fileName = "../../datasets/patient_admissions.csv"
data = read_csv(fileName, parse_dates=[0], index_col=0)

# split into train and test sets
X = data.values
size = int(len(X) * 0.80)
train, test = list(X[0:size]), list(X[size:len(X)])
test_dates = data.index[size:len(X)]

# This will store our predictions, to be compared to real values
predictions = list()

# Walk-forward training and prediction
for t in range(len(test)):
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()

    output = model_fit.forecast()
    pred = output[0]
    predictions.append(pred)
    obs = test[t]
    train.append(obs)
    
    print('Real Value=%.2f, Predicted=%.2f, ' % (obs, pred))
	
# Evaluate predictions
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE of test set: %.3f' % rmse)

# Plot the real values vs predictions
plt.figure(figsize=(10, 6))
test_dates_array = np.array(test_dates)  # Convert to NumPy array
plt.plot(test_dates_array, test, color='blue', label='Actual Values')
plt.plot(test_dates_array, predictions, color='red', label='Predicted Values')
plt.xticks(rotation=45, ha='right')
plt.title('Actual vs Predicted Admissions')
plt.ylabel('Admissions')
plt.legend()
plt.tight_layout()  # Adjust layout
plt.show()
