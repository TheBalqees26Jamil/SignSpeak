import pandas as pd

df = pd.read_csv(
    "data/landmarks_dataset.csv"
)

print("=" * 60)
print("UNIQUE SAMPLE ANALYSIS")
print("=" * 60)

for label in sorted(df["label"].unique()):

    class_df = df[
        df["label"] == label
    ]

    unique_count = len(
        class_df.drop_duplicates()
    )

    total_count = len(
        class_df
    )

    print(
        f"{label:<15} | "
        f"Total: {total_count:>4} | "
        f"Unique: {unique_count:>4}"
    )