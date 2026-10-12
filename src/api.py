
from pathlib import Path
import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Locate files relative to the project root.
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_churn_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"

if not MODEL_PATH.is_file() or not METRICS_PATH.is_file():
    raise FileNotFoundError(
        "Model files are missing. Run python src/evaluate_model.py first."
    )

model = joblib.load(MODEL_PATH)

with METRICS_PATH.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

THRESHOLD = float(metadata["threshold"])
SELECTED_MODEL = metadata["selected_model"]
FEATURES = metadata["features"]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn from customer account information.",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    predicted_churn: Literal["Yes", "No"]
    churn_probability: float
    threshold: float
    model: str


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "selected_model": SELECTED_MODEL,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    # Convert validated input into the feature format used during training.
    customer_data = customer.model_dump()
    customer_df = pd.DataFrame([customer_data])[FEATURES]

    # Match the representation used by the training pipeline.
    customer_df["SeniorCitizen"] = (
        customer_df["SeniorCitizen"].astype("string")
    )

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = "Yes" if probability >= THRESHOLD else "No"

    return PredictionResponse(
        predicted_churn=prediction,
        churn_probability=round(probability, 4),
        threshold=THRESHOLD,
        model=SELECTED_MODEL,
    )
