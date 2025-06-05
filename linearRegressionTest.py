# Import frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

plt.style.use('style_Matplotlib_charts.mplstyle')

# Create a data frame from the csv file
training_data = pd.read_csv('2.3.2.training_data.csv')

x_name = 'BMI'
y_name = 'Target'
x = np.array(training_data[x_name]).reshape(-1, 1)
y = np.array(training_data[y_name])

# Plot the data points
plt.scatter(x, y, marker='x', c='r')
# Set the title
plt.title("Diabetes Disease Progress")
# Set the y-axis label
plt.ylabel(f'Training {training_data[y_name].name}')
# Set the x-axis label
plt.xlabel(f'Training {training_data[x_name].name}')
plt.show()

# Create the model
my_model = LinearRegression()
# Fit the model to the data
my_model.fit(x, y)

y_pred = my_model.predict(x)
plt.plot(x, y_pred)
plt.scatter(x, y, marker='x', c='r')
plt.title("Diabetes Disease Progress")
plt.ylabel(f'Training {training_data[y_name].name}')
plt.xlabel(f'Training {training_data[x_name].name}')
plt.show()

# save the model to disk
filename = 'my_saved_model_v1.sav'
pickle.dump(my_model, open('2.4.Model_Testing_and_Validation/' + filename, 'wb'))






# Create a data frame from the csv file
training_data = pd.read_csv('2.3.2.training_data.csv')

x_name = ['BMI','BP','FDR']
y_name = 'Target'
x = np.array(training_data[x_name])
y = np.array(training_data[y_name])
# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,len(x_name),figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x[:,i],y, label = 'target')
    ax[i].set_xlabel(x_name[i])
ax[0].set_ylabel("Target"); ax[0].legend();
fig.suptitle("Diabetes Disease Progress")
plt.show()
# Create the model
my_model = LinearRegression()
# Fit the model to the data
my_model.fit(x, y)
# scatter plot predictions and targets vs original features    
y_pred = my_model.predict(x)
fig,ax=plt.subplots(1,len(x_name),figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x[:,i],y, label = 'target')
    ax[i].set_xlabel(x_name[i])
    ax[i].scatter(x[:,i],y_pred,color="orange", label = 'predict')
ax[0].set_ylabel("Target"); ax[0].legend();
fig.suptitle("Diabetes Disease Progress")
plt.show()
# save the model to disk
filename = 'my_saved_model_v2.sav'
pickle.dump(my_model, open('2.4.Model_Testing_and_Validation/' + filename, 'wb'))

score = my_model.score(x,y)
print("R^2 score of the model", score)