import pandas as pd

# Load the dataset from your data folder
df = pd.read_csv('data/HateThaiSent.csv')

# 1. See the basic structure and first few rows
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# 2. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. See the distribution of labels (how many hate vs. non-hate messages)
# Replace 'label_column_name' with the actual name of your label column
print("\nLabel Distribution:")
print(df['Hatespeech'].value_counts())