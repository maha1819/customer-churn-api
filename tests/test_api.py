
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200

    result = response.json()
    assert result["predicted_churn"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["threshold"] <= 1
    assert result["model"]


def test_invalid_gender_is_rejected():
    customer = VALID_CUSTOMER.copy()
    customer["gender"] = "Other"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_negative_tenure_is_rejected():
    customer = VALID_CUSTOMER.copy()
    customer["tenure"] = -1

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_missing_required_field_is_rejected():
    customer = VALID_CUSTOMER.copy()
    del customer["Contract"]

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
