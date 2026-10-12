
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = DATA_DIR / "cleaned_churn_median.csv"


def prepare_data() -> None:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    required_columns = {
        "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Convert blank TotalCharges values into missing numeric values.
    raw_charges = df["TotalCharges"].astype("string").str.strip()
    blank_mask = raw_charges.eq("").fillna(False)
    cleaned_charges = raw_charges.mask(blank_mask, pd.NA)

    numeric_charges = pd.to_numeric(cleaned_charges, errors="coerce")
    invalid_mask = (~blank_mask) & numeric_charges.isna()

    if invalid_mask.any():
        examples = raw_charges[invalid_mask].head().tolist()
        raise ValueError(
            f"Unexpected non-numeric TotalCharges values: {examples}"
        )

    df["TotalCharges"] = numeric_charges
    missing_charges = df["TotalCharges"].isna()

    if df.loc[missing_charges, "tenure"].ne(0).any():
        raise ValueError(
            "Missing TotalCharges found for customers with nonzero tenure. "
            "Investigate these records before proceeding."
        )

    if df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique.")

    if not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Unexpected or missing values found in Churn.")

    # Preserve missing values; the ML pipeline handles imputation.
    df.to_csv(OUTPUT_PATH, index=False)

    print("Data preparation completed.")
    print(f"Rows: {len(df):,}")
    print(f"Missing TotalCharges values: {missing_charges.sum()}")
    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare_data()
