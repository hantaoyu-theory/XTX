import pandas as pd

# Change this path if train.csv.gz is not in the same folder
df = pd.read_csv("train.csv.gz", nrows=5)

print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

