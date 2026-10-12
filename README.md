# Customer Churn Prediction API

An end-to-end machine learning project that predicts whether a telecommunications customer is likely to churn. The project covers data cleaning, model training, cross-validation, threshold tuning, REST API development, and automated testing.

## Project Overview

Customer churn occurs when a customer stops using a company's services. Predicting potential churn can help businesses identify customers who may need additional support or retention offers.

This project builds a binary classification pipeline using customer account information and exposes predictions through a REST API built with FastAPI.

## Key Features

- Data inspection, quality checks, and preprocessing
- Comparison of Logistic Regression, Random Forest, and Gradient Boosting
- Stratified train-test splitting and five-fold cross-validation
- Decision-threshold tuning using F1-score
- Evaluation using precision, recall, F1-score, accuracy, ROC-AUC, and a confusion matrix
- FastAPI endpoints for health checks and churn predictions
- Input validation using Pydantic
- Automated API tests using pytest
- Saved model pipeline and evaluation metadata

## Tech Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **API:** FastAPI, Uvicorn
- **Testing:** pytest, FastAPI TestClient
- **Model persistence:** joblib
- **Version control:** Git and GitHub

## Project Structure
customer-churn-api/
├── data/                       # Local datasets, excluded from Git
├── models/
│   ├── best_churn_pipeline.joblib
│   └── model_metrics.json
├── src/
│   ├── inspect_data.py
│   ├── audit_data.py
│   ├── prepare_data.py
│   ├── compare_imputation_strategies.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── tune_models.py
│   └── api.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Model Evaluation

The dataset contains 7,043 customer records, with approximately 26.5% belonging to the churn class. Because the classes are imbalanced, the project evaluates multiple metrics rather than relying on accuracy alone.

Three classification algorithms were compared: Logistic Regression, Random Forest, and Gradient Boosting. Hyperparameters were tuned using randomized search with five-fold stratified cross-validation, using ROC-AUC as the model-selection metric. The decision threshold was tuned using out-of-fold predictions to improve the balance between precision and recall.

**Selected model:** Gradient Boosting  
**Decision threshold:** 0.31  
**Model-selection metric:** Five-fold cross-validation ROC-AUC

| Metric | Held-out test result |
|---|---:|
| Accuracy | 76.72% |
| Balanced accuracy | 76.30% |
| Precision | 54.44% |
| Recall | 75.40% |
| F1-score | 63.23% |
| ROC-AUC | 84.63% |

Recall measures the proportion of actual churners identified by the model, while precision measures how often its positive churn predictions are correct. The decision threshold affects this trade-off and can be adjusted to suit business priorities.

These metrics are development results from the current held-out test split. Since the test results have been examined during development, evaluation on a new, untouched dataset would provide a stronger estimate of generalization performance.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/maha1819/customer-churn-api.git
cd customer-churn-api
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The trained model and its metadata are included in the repository, so you do not need to train a model before starting the API.

## Run the API

From the project root, run:

```bash
python -m uvicorn src.api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict customer churn |

### Example Prediction Request

Send a JSON request to `POST /predict`:

```json
{
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
  "TotalCharges": 29.85
}
```

The response includes the predicted churn label, churn probability, decision threshold, and selected model.

## Run Automated Tests

Run the following command from the project root:

```bash
python -m pytest -v
```

The current test suite covers successful API responses, prediction output validation, and rejection of invalid or incomplete inputs.

## Future Improvements

- Add Docker support for reproducible deployment
- Add continuous integration to run tests automatically
- Deploy the API to a cloud hosting platform
- Build a Streamlit interface for interactive predictions
- Add model monitoring and drift detection
- Evaluate performance on a new, untouched test dataset

## Disclaimer

This project is intended for educational and portfolio purposes. Predictions are estimates based on historical data and should not be treated as guaranteed outcomes.
