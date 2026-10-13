# Customer Churn Prediction API

An end-to-end machine learning project that predicts whether a telecommunications customer is likely to churn. The project covers data preprocessing, model comparison, cross-validation, decision-threshold tuning, REST API development, an interactive Streamlit frontend, automated testing, and Docker-based deployment.

## Project Overview

Customer churn occurs when a customer stops using a company's services. Predicting potential churn helps businesses identify customers who may need additional support or retention offers.

This project builds a binary classification pipeline using customer account information. It exposes predictions through a REST API built with FastAPI and provides an interactive interface using Streamlit.

## Key Features

- Data inspection, quality checks, and preprocessing
- Comparison of Logistic Regression, Random Forest, and Gradient Boosting
- Stratified train-test splitting and five-fold cross-validation
- Hyperparameter tuning using randomized search
- Decision-threshold tuning to improve churn recall
- Evaluation using accuracy, balanced accuracy, precision, recall, F1-score, and ROC-AUC
- FastAPI endpoints for health checks and churn predictions
- Pydantic input validation
- Interactive Streamlit frontend for customer churn predictions
- Automated API tests using pytest
- Model persistence using joblib
- Separate Docker images for the API and frontend
- Docker Compose orchestration for running both services together

## Tech Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **API:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Validation:** Pydantic
- **Model persistence:** joblib
- **Testing:** pytest, FastAPI TestClient
- **Containerization:** Docker, Docker Compose
- **Version control:** Git and GitHub

## Project Structure

```text
customer-churn-api/
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
│   ├── api.py
│   └── streamlit_app.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── requirements-streamlit.txt
├── .dockerignore
├── .gitignore
└── README.md
```

The local dataset directory is excluded from Git. The trained model pipeline and evaluation metadata are included in the repository.

## Model Development and Evaluation

The dataset contains 7,043 customer records, with approximately 26.5% belonging to the churn class. Because the classes are imbalanced, multiple evaluation metrics are used instead of relying on accuracy alone.

Three classification algorithms were compared: Logistic Regression, Random Forest, and Gradient Boosting. Hyperparameters were tuned using randomized search with five-fold stratified cross-validation, with ROC-AUC as the model-selection metric. The decision threshold was tuned using out-of-fold predictions to improve the balance between precision and recall.

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

Recall measures the proportion of actual churners identified by the model, while precision measures how often positive churn predictions are correct. The decision threshold affects this trade-off and can be adjusted to suit business priorities.

These metrics are development results from the current held-out test split. Evaluation on a new, untouched dataset would provide a stronger estimate of generalization performance.

## Run Locally

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

### 3. Install API dependencies

```bash
python -m pip install -r requirements.txt
```

The trained model and its metadata are included in the repository, so model training is not required to start the API.

### 4. Start the API

```bash
python -m uvicorn src.api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

### Available API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict customer churn |

The prediction endpoint accepts customer information and returns the predicted churn label, churn probability, decision threshold, and selected model.

## Run the Streamlit Frontend Locally

Start the FastAPI backend first. Open a second terminal, activate the same virtual environment, and install the frontend dependencies:

```bash
python -m pip install -r requirements-streamlit.txt
```

Start the Streamlit application:

```bash
python -m streamlit run src/streamlit_app.py
```

Open the frontend:

http://localhost:8501

Enter the customer's details and submit the form to view the predicted churn result and probability.

When running outside Docker, the frontend connects to the API at `http://127.0.0.1:8000/predict`.

## Run with Docker

### Prerequisites

- Docker Desktop installed and running
- Git

### Option 1: Run the API only

Build the API image:

```bash
docker build -t customer-churn-api .
```

Start the API container:

```bash
docker run --rm -p 8000:8000 --name churn-api-container customer-churn-api
```

Open the health endpoint:

http://localhost:8000/health

Open the API documentation:

http://localhost:8000/docs

Stop the container using `Ctrl+C`.

### Option 2: Run the API and frontend with Docker Compose

Docker Compose builds and runs the API and Streamlit frontend as separate services.

From the project root, run:

```bash
docker compose up --build
```

This command builds the images if needed and starts both services.

**Access the applications:**

- **Streamlit frontend:** http://localhost:8501
- **FastAPI documentation:** http://localhost:8000/docs
- **API health check:** http://localhost:8000/health

The Streamlit container communicates with the API using the internal Docker Compose service address `http://api:8000/predict`. The frontend is available through host port `8501`, while the API is available through host port `8000`.

To stop the services, press `Ctrl+C` and run:

```bash
docker compose down
```

## Run Automated Tests

Install the testing dependencies in your active virtual environment if they are not already installed:

```bash
python -m pip install pytest httpx
```

Run the tests from the project root:

```bash
python -m pytest -v
```

The API test suite covers successful responses, prediction output validation, and rejection of invalid or incomplete inputs.

## Future Improvements

- Add continuous integration to run tests automatically
- Deploy the application to a cloud hosting platform
- Add model monitoring and data drift detection
- Evaluate performance on a new, untouched dataset
- Explore explainability techniques to help interpret individual predictions

## Disclaimer

Predictions are estimates based on historical data and should not be treated as guaranteed outcomes.

