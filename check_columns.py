import pandas as pd

# Load without column names
df = pd.read_csv('data/KDDTrain+.csv', header=None)
print(f"Total columns: {df.shape[1]}")
print(f"Last column (index {df.shape[1]-1}) sample values:")
print(df[df.shape[1]-1].value_counts().head())

# Column 41 should be the label
print(f"\nColumn 41 sample values:")
print(df[41].value_counts().head())
