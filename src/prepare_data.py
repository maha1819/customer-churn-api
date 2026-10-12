
from pathlib import Path

import pandas as pd


# Resolve paths relative to the project, not the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise FileNotFoundError(
        f"Expected exactly one raw CSV in {DATA_DIR}; found {len(csv_files)}."
    )

df = pd.read_csv(csv_files[0])

# 1. Validate the expected schema before modifying the data.
required_columns = {
    "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# 2. Clean TotalCharges and convert it to a numeric type.
raw_charges = df["TotalCharges"].astype("string").str.strip()
blank_mask = raw_charges.eq("").fillna(False)

cleaned_charges = raw_charges.mask(blank_mask, pd.NA)
df["TotalCharges"] = pd.to_numeric(cleaned_charges, errors="coerce")

# 3. Validate that missing charges have the pattern found in our audit.
missing_charges = df["TotalCharges"].isna()

if df.loc[missing_charges, "tenure"].ne(0).any():
    raise ValueError(
        "Found missing TotalCharges for customers whose tenure is not zero. "
        "Investigate these records before proceeding."
    )

# 4. Validate IDs and target labels.
if df["customerID"].isna().any() or df["customerID"].duplicated().any():
    raise ValueError("Customer IDs must be present and unique.")

if not df["Churn"].isin(["Yes", "No"]).all():
    raise ValueError("Unexpected or missing values found in Churn.")

# 5. Save a cleaned local copy. The original CSV remains unchanged.
output_path = DATA_DIR / "cleaned_churn.csv"
df.to_csv(output_path, index=False)

print("Data preparation completed.")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"TotalCharges dtype: {df['TotalCharges'].dtype}")
print(f"Missing TotalCharges: {df['TotalCharges'].isna().sum()}")
print(f"Unique customer IDs: {df['customerID'].nunique():,}")
print(f"Churn rate: {df['Churn'].eq('Yes').mean():.2%}")
print(f"Cleaned file saved to: {output_path}")
