
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_churn_median.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        "Cleaned data not found. Run python src/prepare_data.py first."
    )

df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop(columns=["customerID", "Churn"]).copy()
X["SeniorCitizen"] = X["SeniorCitizen"].astype("string")
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

# Re-create preprocessing inside this script.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

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

# Reserve a test set. Do not use it to choose the model or threshold.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
candidates = {}

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(classifier)),
    ])

    # Each training row receives a probability from a model that
    # did not train on that row.
    oof_probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # Choose the threshold that gives the best F1 on out-of-fold
    # training predictions, not on the test set.
    thresholds = [i / 100 for i in range(20, 81, 5)]
    threshold_scores = []

    for threshold in thresholds:
        oof_predictions = (oof_probabilities >= threshold).astype(int)
        score = f1_score(
            y_train, oof_predictions, zero_division=0
        )
        threshold_scores.append((score, threshold))

    best_cv_f1, best_threshold = max(threshold_scores)

    # Fit on all training data; evaluate once on the held-out test set.
    fitted_pipeline = clone(pipeline).fit(X_train, y_train)
    test_probabilities = fitted_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    metrics = {
        "model": name,
        "oof_f1_at_selected_threshold": float(best_cv_f1),
        "threshold": float(best_threshold),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_precision": float(precision_score(
            y_test, test_predictions, zero_division=0
        )),
        "test_recall": float(recall_score(
            y_test, test_predictions, zero_division=0
        )),
        "test_f1": float(f1_score(
            y_test, test_predictions, zero_division=0
        )),
        "test_roc_auc": float(roc_auc_score(
            y_test, test_probabilities
        )),
    }

    results.append(metrics)
    candidates[name] = fitted_pipeline

    print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
    for key, value in metrics.items():
        if key != "model":
            print(f"{key}: {value:.4f}")

    print("\nTest confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))
    print("\nTest classification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

# Select by cross-validated F1, not by test F1.
results_df = pd.DataFrame(results).sort_values(
    "oof_f1_at_selected_threshold", ascending=False
)
print("\nCROSS-VALIDATION MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_metrics = next(row for row in results if row["model"] == best_name)

# Refit the chosen model on the entire training set.
best_pipeline = candidates[best_name]
joblib.dump(best_pipeline, MODEL_DIR / "best_churn_pipeline.joblib")

metadata = {
    "selected_model": best_name,
    "threshold": best_metrics["threshold"],
    "selection_metric": "out-of-fold training F1",
    "cv_folds": 5,
    "random_state": 42,
    "features": list(X.columns),
    "positive_class": "Yes",
    "test_metrics": {
        key: value
        for key, value in best_metrics.items()
        if key.startswith("test_")
    },
    "all_model_results": results,
}

with (MODEL_DIR / "model_metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print(f"\nSelected using cross-validation: {best_name}")
print(f"Chosen threshold: {best_metrics['threshold']:.2f}")
print(f"Saved pipeline: {MODEL_DIR / 'best_churn_pipeline.joblib'}")
print(f"Saved metadata: {MODEL_DIR / 'model_metrics.json'}")
