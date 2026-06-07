import pandas as pd

DATASET_PATH = "data/landmarks_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nNumber of Features:")
print(len(df.columns) - 1)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nNumber of Classes:")
print(df["label"].nunique())

print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Check Completed Successfully.")