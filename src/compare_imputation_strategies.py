from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATASETS = {
    "Median imputation": DATA_DIR / "cleaned_churn_median.csv",
    "Zero fill": DATA_DIR / "cleaned_churn_zero.csv",
}

RANDOM_STATE = 42
THRESHOLDS = np.arange(0.20, 0.801, 0.05)


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def main():
    results = []

    for strategy, path in DATASETS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Run src/prepare_data.py first."
            )

        df = pd.read_csv(path)

        if df["TotalCharges"].isna().any():
            if strategy != "Median imputation":
                raise ValueError(f"Unexpected missing charges in {strategy}.")

        X = df.drop(columns=["customerID", "Churn"])
        y = df["Churn"].map({"No": 0, "Yes": 1})

        # Keep categorical encoding consistent with the existing project.
        X["SeniorCitizen"] = X["SeniorCitizen"].astype(str)

        numeric_features = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]
        categorical_features = [
            column for column in X.columns
            if column not in numeric_features
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        for model_name, estimator in build_models().items():
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ])

            # Tune the decision threshold using training-set
            # out-of-fold predictions only.
            oof_probabilities = cross_val_predict(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]

            best_threshold = 0.50
            best_oof_f1 = -1.0

            for threshold in THRESHOLDS:
                oof_predictions = (
                    oof_probabilities >= threshold
                ).astype(int)

                score = f1_score(
                    y_train,
                    oof_predictions,
                    zero_division=0,
                )

                if score > best_oof_f1:
                    best_oof_f1 = score
                    best_threshold = float(threshold)

            # Fit on training data and evaluate on the held-out test set.
            pipeline.fit(X_train, y_train)
            test_probabilities = pipeline.predict_proba(X_test)[:, 1]
            test_predictions = (
                test_probabilities >= best_threshold
            ).astype(int)

            results.append({
                "Strategy": strategy,
                "Model": model_name,
                "Threshold": best_threshold,
                "CV F1": best_oof_f1,
                "Test Accuracy": accuracy_score(y_test, test_predictions),
                "Test Balanced Accuracy": balanced_accuracy_score(
                    y_test, test_predictions
                ),
                "Test Precision": precision_score(
                    y_test, test_predictions, zero_division=0
                ),
                "Test Recall": recall_score(
                    y_test, test_predictions, zero_division=0
                ),
                "Test F1": f1_score(
                    y_test, test_predictions, zero_division=0
                ),
                "Test ROC-AUC": roc_auc_score(
                    y_test, test_probabilities
                ),
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["Strategy", "CV F1"],
        ascending=[True, False],
    )

    print("\nMissing-value strategy comparison")
    print("=" * 100)
    print(results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()