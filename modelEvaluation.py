
# Import frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('Nappa-Theft/style_Matplotlib_charts.mplstyle')
from sklearn.linear_model import LinearRegression
import pickle


filename = 'Nappa-Theft/2.4.Model_Testing_and_Validation/my_saved_model_v1.sav'
model_A = pickle.load(open(filename, 'rb'))


x_col = 'BMI'
testing_data = pd.read_csv('Nappa-Theft/2.4.1.testing_data.csv')
x_test = np.array(testing_data[x_col]).reshape(-1,1)
y_test = np.array(testing_data['Target'])

test_score = model_A.score(x_test, y_test)
print(f'Training data score: {test_score}')

table = pd.DataFrame({
    testing_data.columns[0]: x_test.flatten(),  # Flatten x for easy display
    testing_data.columns[1]: y_test,
    'Predicted result':model_A.predict(x_test),
    'Loss' : abs(model_A.predict(x_test).round(2) - y_test)**2
})
print(table)
cost = 1 / (2 * table.shape[0]) * table['Loss'].sum()

print(f"The cost or average loss of this model is {cost}")


print(f'X Axis intercept: {model_A.intercept_}')
print(f'Coefficient: {model_A.coef_}')

score = model_A.score(x_test, y_test)
print("R^2 score of the model", score)