import pandas as pd

file_path = "data/raw/parkinsons.data"

dataset = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print()

print("Rows and Columns:")
print(dataset.shape)

print()

print("First 5 Rows:")
print(dataset.head())