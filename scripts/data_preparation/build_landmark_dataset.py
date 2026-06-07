import os
import pandas as pd

INPUT_FOLDER = "landmarks_raw"
OUTPUT_FOLDER = "data"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

all_dataframes = []

for file in os.listdir(INPUT_FOLDER):

    if file.endswith(".csv"):

        file_path = os.path.join(INPUT_FOLDER, file)

        df = pd.read_csv(file_path)

        all_dataframes.append(df)

        print(f"Loaded: {file} ({len(df)} samples)")

merged_df = pd.concat(
    all_dataframes,
    ignore_index=True
)

output_path = os.path.join(
    OUTPUT_FOLDER,
    "landmarks_dataset.csv"
)

merged_df.to_csv(
    output_path,
    index=False
)

print("\n" + "="*40)
print("Dataset Merged Successfully")
print("="*40)
print("Total Samples :", len(merged_df))
print("Total Features:", len(merged_df.columns)-1)
print("Total Classes :", merged_df["label"].nunique())
print(f"Saved To      : {output_path}")