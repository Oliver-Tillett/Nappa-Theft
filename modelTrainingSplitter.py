import pandas as pd 
from sklearn.model_selection import train_test_split

# Create a data frame from the csv file
data = pd.read_csv('2.3.1.model_ready_data.csv')

# Split the data into training and testing data
training_data, testing_data = train_test_split(data, test_size=0.25)

# Save the training and testing data to csv files
training_data.to_csv('2.3.2.training_data.csv', index=False)
testing_data.to_csv('2.4.1.testing_data.csv', index=False)