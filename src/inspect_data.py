
from pathlib import Path

import pandas as pd


# Locate the dataset relative to this script, not the terminal's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise FileNotFoundError(
        f"Expected exactly one CSV in {DATA_DIR}, "
        f"but found {len(csv_files)}."
    )

csv_path = csv_files[0]
df = pd.read_csv(csv_path)

print(f"\nDataset loaded: {csv_path.name}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n--- First 5 rows ---")
print(df.head().to_string(index=False))

print("\n--- Column data types ---")
print(df.dtypes.to_string())

print("\n--- Missing values per column ---")
missing = df.isna().sum()
print(missing[missing > 0].to_string() if missing.any() else "No pandas-recognized missing values.")

print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

if "Churn" in df.columns:
    print("\n--- Churn target distribution ---")
    print(df["Churn"].value_counts(dropna=False).to_string())
    print("\n--- Churn proportions ---")
    print(
        df["Churn"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )
else:
    print("\nWARNING: No column named 'Churn' was found.")
