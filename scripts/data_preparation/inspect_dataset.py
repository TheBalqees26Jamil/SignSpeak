import pandas as pd

df = pd.read_csv(
    "data/landmarks_dataset.csv"
)

print("=" * 60)
print("DATASET INFO")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nFeature Count:")
print(df.shape[1] - 1)