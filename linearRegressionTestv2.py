import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('Nappa-Theft/style_Matplotlib_charts.mplstyle')
from sklearn.linear_model import LinearRegression
import pickle #For saving the trained model.

# Create a data frame from the csv file
training_data = pd.read_csv('Nappa-Theft/2.3.2.training_data.csv')

x_name = 'BP'
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

# Create the model
my_model = LinearRegression()
# Fit the model to the data
my_model.fit(x, y)

y_pred = my_model.predict(x)
print("Predicted vs Actual (first 10):")

for actual, predicted in list(zip(y, y_pred))[:10]: 
    #zip(y, y_pred) pairs each actual and predicted value as a tuple.
    print(f"Actual: {actual:.2f}  |  Predicted: {predicted:.2f}")

# Plot the data points as red, x, marks
plt.scatter(x, y, marker='x', c='r')
#color of marker,x, is red

# Set the title
plt.title("Diabetes Disease Progress")
# Set the y-axis label
plt.ylabel(f'Training {training_data[y_name].name}')
# Set the x-axis label
plt.xlabel(f'Training {training_data[x_name].name}')
plt.show()

y_pred = my_model.predict(x)
print("Predicted vs Actual (first 10):")

for actual, predicted in list(zip(y, y_pred))[:10]: 
    #zip(y, y_pred) pairs each actual and predicted value as a tuple.
    print(f"Actual: {actual:.2f}  |  Predicted: {predicted:.2f}")

# R² score: how much variance is explained
score = my_model.score(x, y)
print("R^2 score of the model", score)