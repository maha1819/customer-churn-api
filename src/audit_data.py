
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise FileNotFoundError(
        f"Expected exactly one CSV in {DATA_DIR}; found {len(csv_files)}."
    )

df = pd.read_csv(csv_files[0])

print("\n=== 1. IDENTIFIER CHECK ===")
print("Unique customer IDs:", df["customerID"].nunique())
print("Total rows:", len(df))
print("Missing customer IDs:", df["customerID"].isna().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

print("\n=== 2. TARGET CHECK ===")
print("Churn labels:")
print(df["Churn"].value_counts(dropna=False))
print("Unexpected Churn labels:")
print(df.loc[~df["Churn"].isin(["Yes", "No"]), "Churn"].value_counts(dropna=False))

print("\n=== 3. WHITESPACE CHECK ===")
for col in df.select_dtypes(include=["object", "str", "string"]).columns:
    values = df[col].astype("string")
    nonmissing = values.notna()
    whitespace_only = (values.str.strip() == "").fillna(False)

    leading_or_trailing = (
        values.ne(values.str.strip()).fillna(False)
    )

    if whitespace_only.any() or leading_or_trailing.any():
        print(
            f"{col}: whitespace-only={int(whitespace_only.sum())}, "
            f"leading/trailing whitespace={int(leading_or_trailing.sum())}"
        )

print("\n=== 4. CATEGORICAL VALUES ===")
categorical_cols = [
    col for col in df.select_dtypes(
        include=["object", "str", "string"]
    ).columns
    if col not in ["customerID", "TotalCharges"]
]

for col in categorical_cols:
    print(f"\n{col}: {df[col].nunique(dropna=False)} distinct values")
    print(df[col].value_counts(dropna=False).to_dict())

print("\n=== 5. NUMERIC RANGES ===")
for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
    print(f"\n{col}:")
    print(df[col].describe().to_string())

print("\nUnexpected SeniorCitizen values:")
print(df.loc[~df["SeniorCitizen"].isin([0, 1]), "SeniorCitizen"].value_counts())

print("\nNegative tenure values:", int((df["tenure"] < 0).sum()))
print("Negative MonthlyCharges:", int((df["MonthlyCharges"] < 0).sum()))

print("\n=== 6. TOTALCHARGES CHECK ===")
raw = df["TotalCharges"].astype("string")
cleaned = raw.str.strip().replace("", pd.NA)
charges = pd.to_numeric(cleaned, errors="coerce")

blank = cleaned.isna()
invalid = cleaned.notna() & charges.isna()

print("Blank TotalCharges:", int(blank.sum()))
print("Nonblank but nonnumeric TotalCharges:", int(invalid.sum()))
print("Negative numeric TotalCharges:", int((charges < 0).sum()))

print("\nBlank charges by tenure:")
print(df.loc[blank, "tenure"].value_counts(dropna=False).to_dict())

print("\nZero-tenure customers:", int((df["tenure"] == 0).sum()))
print("Zero-tenure customers with blank charges:",
      int(((df["tenure"] == 0) & blank).sum()))
