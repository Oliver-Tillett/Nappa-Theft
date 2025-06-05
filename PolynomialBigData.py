import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.style.use('Nappa-Theft/style_Matplotlib_charts.mplstyle')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle 

training_data = pd.read_csv('Nappa-Theft/2.3.2.training_data.csv')
x_name = ['BMI']
y_name = 'Target'
x = np.array(training_data[x_name])
y = np.array(training_data[y_name])

if(len(x_name) == 1):
    plt.scatter(x, y, marker='x', c='r')
    plt.title("Diabetes Disease Progress")
    plt.ylabel(f'Training {y_name}')
    plt.xlabel(f'Training {x_name[0]}')
else:
    fig,ax=plt.subplots(1,len(x_name),figsize=(12,3))
    for i in range(len(ax)):
        ax[i].scatter(x[:,i],y, label = 'target')
        ax[i].set_xlabel(x_name[i])
    ax[0].set_ylabel("Target"); ax[0].legend();
    fig.suptitle("Diabetes Disease Progress")
plt.show()