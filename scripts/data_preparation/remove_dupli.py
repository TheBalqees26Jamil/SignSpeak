import pandas as pd



df = pd.read_csv(
    "data/landmarks_dataset.csv"
)

original_samples = len(df)



df_clean = df.drop_duplicates()

clean_samples = len(df_clean)

removed_samples = (
    original_samples -
    clean_samples
)



output_path = (
    "data/landmarks_dataset_clean.csv"
)

df_clean.to_csv(
    output_path,
    index=False
)



print("=" * 50)
print("DATASET CLEANING REPORT")
print("=" * 50)

print(
    f"Original Samples : {original_samples}"
)

print(
    f"Clean Samples    : {clean_samples}"
)

print(
    f"Removed Samples  : {removed_samples}"
)

print("\nSaved To:")
print(output_path)

print("\nDone Successfully.")


