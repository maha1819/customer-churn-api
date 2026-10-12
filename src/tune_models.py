

from pathlib import Path
import json

import joblib
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
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_churn_median.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

RANDOM_STATE = 42
CV_SPLITS = 5
N_ITER = 8


def build_pipeline(model):
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. "
            "Run python src/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected or missing churn labels.")

    # Preserve the project's existing stratified train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Match the categorical representation used by the API.
    X_train["SeniorCitizen"] = X_train["SeniorCitizen"].astype(str)
    X_test["SeniorCitizen"] = X_test["SeniorCitizen"].astype(str)

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.7],
                "model__class_weight": [
                    None,
                    "balanced",
                    "balanced_subsample",
                ],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.1, 0.15],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_leaf": [1, 3, 5],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        ),
    }

    results = []
    candidates = {}

    for name, (estimator, param_grid) in models.items():
        print(f"\n{'=' * 60}")
        print(f"Tuning: {name}")
        print("=" * 60)

        search = RandomizedSearchCV(
            estimator=build_pipeline(estimator),
            param_distributions=param_grid,
            n_iter=N_ITER,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Use out-of-fold training predictions to choose the threshold.
        train_probabilities = cross_val_predict(
            clone(best_model),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        thresholds = np.arange(0.20, 0.81, 0.01)

        threshold_scores = [
            f1_score(
                y_train,
                (train_probabilities >= threshold).astype(int),
                zero_division=0,
            )
            for threshold in thresholds
        ]

        best_index = int(np.argmax(threshold_scores))
        best_threshold = float(thresholds[best_index])

        # Development evaluation on the existing held-out test split.
        test_probabilities = best_model.predict_proba(X_test)[:, 1]
        test_predictions = (
            test_probabilities >= best_threshold
        ).astype(int)

        metrics = {
            "model": name,
            "cv_roc_auc": float(search.best_score_),
            "threshold": best_threshold,
            "accuracy": float(accuracy_score(y_test, test_predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, test_predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, test_predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, test_predictions, zero_division=0
                )
            ),
            "f1": float(
                f1_score(y_test, test_predictions, zero_division=0)
            ),
            "test_roc_auc": float(
                roc_auc_score(y_test, test_probabilities)
            ),
        }

        results.append(metrics)

        candidates[name] = {
            "model": best_model,
            "threshold": best_threshold,
            "metrics": metrics,
            "best_params": search.best_params_,
        }

        print("Best parameters:")
        print(search.best_params_)
        print(f"Selected threshold: {best_threshold:.2f}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Test precision: {metrics['precision']:.4f}")
        print(f"Test recall: {metrics['recall']:.4f}")
        print(f"Test F1-score: {metrics['f1']:.4f}")
        print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        by="cv_roc_auc",
        ascending=False,
    )

    print("\n\nFINAL COMPARISON")
    print(results_df.round(4).to_string(index=False))

    # Select the model with the highest cross-validation ROC-AUC.
    selected_name = max(
        candidates,
        key=lambda name: candidates[name]["metrics"]["cv_roc_auc"],
    )

    selected = candidates[selected_name]
    selected_metrics = selected["metrics"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the fitted pipeline and matching metadata.
    joblib.dump(selected["model"], MODEL_PATH)

    metadata = {
        "selected_model": selected_name,
        "threshold": selected["threshold"],
        "selection_metric": "5-fold cross-validation ROC-AUC",
        "cv_folds": CV_SPLITS,
        "random_state": RANDOM_STATE,
        "features": list(X.columns),
        "positive_class": "Yes",
        "best_params": selected["best_params"],
        "test_metrics": {
            "test_accuracy": selected_metrics["accuracy"],
            "test_balanced_accuracy": selected_metrics["balanced_accuracy"],
            "test_precision": selected_metrics["precision"],
            "test_recall": selected_metrics["recall"],
            "test_f1": selected_metrics["f1"],
            "test_roc_auc": selected_metrics["test_roc_auc"],
        },
        "all_model_results": results,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nSelected model: {selected_name}")
    print(f"Selected threshold: {selected['threshold']:.2f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
