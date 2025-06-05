import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('Nappa-Theft/style_Matplotlib_charts.mplstyle')
from sklearn.linear_model import LinearRegression
import pickle

# Create a data frame from the csv file
training_data = pd.read_csv('Nappa-Theft/2.3.2.training_data.csv')
x_name = ['BMI','BP','FDR']
y_name = 'Target'
x = np.array(training_data[x_name])
y = np.array(training_data[y_name])

# scatter plot original features vs actual output    
fig,ax=plt.subplots(1,len(x_name),figsize=(12,3),sharey=True)
#plt.subplots creates multiple subplots in one figure
#plt.subplots(1,len(x_name) one row of plots
#sharey=True All subplots share the same Y-axis scale
#ax represents an array of axes objects

for i in range(len(ax)):
    ax[i].scatter(x[:,i],y, label = 'target')
    ax[i].set_xlabel(x_name[i])
ax[0].set_ylabel("Target"); ax[0].legend()
fig.suptitle("Acutals\nBMI-Target, BP-Target, FDR-Target")
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
ax[0].set_ylabel("Target"); ax[0].legend()
fig.suptitle("Predictions\nTarget-BMI, Target-BP, Target-FDR")
plt.show()

score = my_model.score(x, y)
print("R^2 score of the model", score)