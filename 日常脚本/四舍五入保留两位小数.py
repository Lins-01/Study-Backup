import pandas as pd

# Define the path to the CSV file
csv_file_path = './value.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(csv_file_path)

# Round the 'value' column to two decimal places
data['value'] = data['value'].round(2)

# Write the modified DataFrame back to the same CSV file, overwriting it
data.to_csv(csv_file_path, index=False)
