
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.base import clone
# 1. Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_churn_median.csv"
MODEL_DIR = PROJECT_ROOT / "models"

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}. "
        "Run python src/prepare_data.py first."
    )

# 2. Load data and validate it
df = pd.read_csv(DATA_PATH)

required_columns = {
    "customerID", "Churn", "tenure",
    "MonthlyCharges", "TotalCharges"
}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df["customerID"].isna().any() or df["customerID"].duplicated().any():
    raise ValueError("Customer IDs must be present and unique.")

if not df["Churn"].isin(["Yes", "No"]).all():
    raise ValueError("Churn must contain only Yes and No.")

# 3. Separate features (X) from the target (y)
# CustomerID is an identifier, not a predictive feature.
X = df.drop(columns=["customerID", "Churn"]).copy()
y = df["Churn"].map({"No": 0, "Yes": 1})

# Treat SeniorCitizen as a category rather than a continuous measurement.
X["SeniorCitizen"] = X["SeniorCitizen"].astype("string")

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

# 4. Split before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")
print(f"Features: {X.shape[1]}")
print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {len(categorical_features)}")

# 5. Build preprocessing steps
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 6. Define the models to compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
    ),
}

results = []
fitted_models = {}

# 7. Train and evaluate each complete pipeline
for name, classifier in models.items():
    pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", classifier),
])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)
    fitted_models[name] = pipeline

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 8. Compare models and select the highest churn F1 score
results_df = pd.DataFrame(results).sort_values(
    by="f1", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_pipeline = fitted_models[best_name]

# 9. Save the complete pipeline and evaluation results
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "best_churn_pipeline.joblib"
metrics_path = MODEL_DIR / "model_metrics.json"

joblib.dump(best_pipeline, model_path)

with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print(f"\nSelected model by churn F1: {best_name}")
print(f"Saved pipeline to: {model_path}")
print(f"Saved metrics to: {metrics_path}")
