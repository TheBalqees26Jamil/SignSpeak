import os
import pandas as pd

INPUT_FOLDER = "data/landmarks_raw"
OUTPUT_FILE = "data/landmarks_dataset.csv"

all_dataframes = []

files = sorted(os.listdir(INPUT_FOLDER))

for file in files:

    if not file.endswith(".csv"):
        continue

    path = os.path.join(
        INPUT_FOLDER,
        file
    )

    df = pd.read_csv(path)

    all_dataframes.append(df)

    print(
        f"{file:<25} {len(df)} samples"
    )

dataset = pd.concat(
    all_dataframes,
    ignore_index=True
)

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("Samples :", len(dataset))
print("Features:", dataset.shape[1] - 1)

print("\nSaved To:")
print(OUTPUT_FILE)