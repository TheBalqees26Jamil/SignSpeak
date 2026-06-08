import pandas as pd



df = pd.read_csv(
    "data/landmarks_dataset.csv"
)

print("=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

results = []

total_duplicates = 0



for label in sorted(df["label"].unique()):

    class_df = df[
        df["label"] == label
    ]

    duplicates = class_df.duplicated().sum()

    total = len(class_df)

    duplicate_percent = (
        duplicates / total
    ) * 100

    total_duplicates += duplicates

    results.append([
        label,
        total,
        duplicates,
        round(
            duplicate_percent,
            2
        )
    ])



result_df = pd.DataFrame(
    results,
    columns=[
        "Class",
        "Total Samples",
        "Duplicates",
        "Duplicate %"
    ]
)

result_df = result_df.sort_values(
    by="Duplicates",
    ascending=False
)

print(
    result_df.to_string(
        index=False
    )
)

print("\n" + "=" * 60)

print(
    f"\nTotal Dataset Duplicates: {total_duplicates}"
)