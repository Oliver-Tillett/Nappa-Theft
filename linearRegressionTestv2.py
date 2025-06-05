import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('style_Matplotlib_charts.mplstyle')
from sklearn.linear_model import LinearRegression
import pickle #For saving the trained model.

# Create a data frame from the csv file
training_data = pd.read_csv('2.3.2.training_data.csv')

x_name = 'BMI'
y_name = 'Target'
x = np.array(training_data[x_name]).reshape(-1, 1) 
#reshape() makes x into a 2D array (required by LinearRegression.fit()).
#-1 argument indicates “Figure out the number of rows automatically based 
#   on the length of the array.”
#1 argument value indicates “Make it a column vector with 1 column.”
y = np.array(training_data[y_name])
print("First 10 x and y values:")
print("x (BMI):", x[:10].flatten())  # flatten() to make it 1D for readability
print("y (Target):", y[:10])
