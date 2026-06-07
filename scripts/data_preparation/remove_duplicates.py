import pandas as pd
import os

INPUT_FILE = "data/landmarks_dataset.csv"
OUTPUT_FILE = "data/landmarks_dataset_clean.csv"

df = pd.read_csv(INPUT_FILE)

original_count = len(df)

df_clean = df.drop_duplicates()

clean_count = len(df_clean)

removed = original_count - clean_count

df_clean.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 50)
print("DATASET CLEANING REPORT")
print("=" * 50)

print(f"Original Samples : {original_count}")
print(f"Clean Samples    : {clean_count}")
print(f"Removed Samples  : {removed}")

print("\nSaved To:")
print(OUTPUT_FILE)

print("\nDone Successfully.")