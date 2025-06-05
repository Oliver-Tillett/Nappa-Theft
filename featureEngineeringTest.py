import pandas as pd

data_frame = pd.read_csv("2.2.1.wrangled_data.csv")

data_frame['SEX'] = data_frame['SEX'].apply(lambda gender: -1 if gender.lower() == 'male' else 1 if gender.lower() == 'female' else None)
print(data_frame['SEX'].head())

# Convert the 'DoB' and 'DoTest' columns to datetime
data_frame['DoB'] = pd.to_datetime(data_frame['DoB'], format='%d/%m/%Y')
data_frame['DoT'] = pd.to_datetime(data_frame['DoT'], format='%d/%m/%Y')

# Calculate the year difference
data_frame['Age'] = ((data_frame['DoT'] - data_frame['DoB']).dt.days  / 365.25).round()

# Print the result
print(data_frame[['DoB', 'DoT', 'Age']].head())

# Calculate the year difference and round to an integer
data_frame['Age'] = ((data_frame['DoT'] - data_frame['DoB']).dt.days / 365.25).round().astype(int)

# Create the 'Risk' column
data_frame['Risk'] = data_frame['BMI'] * data_frame['Age']

# Calculate the percentage of the maximum risk
data_frame['Risk%'] = (data_frame['Risk'] / data_frame['Risk'].max()).round(2)

# Print the result
print(data_frame[['Age', 'BMI', 'Risk%']])

# Calculate the family history risk
data_frame['FHRisk'] = (data_frame['FDR'] / data_frame['FDR'].max())

# Scale the result between 0.15 and 0.85
min_val = 0.15
max_val = 0.85
data_frame['FHRisk'] = (((data_frame['FHRisk'] - data_frame['FHRisk'].min()) / (data_frame['FHRisk'].max() - data_frame['FHRisk'].min())) * (max_val - min_val) + min_val).round(2)

# Print the result
print(data_frame[['Age', 'FDR', 'FHRisk']])

data_frame.to_csv('2.3.1.model_ready_data.csv', index=False)