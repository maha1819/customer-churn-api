

import os

import requests
import streamlit as st


# 1. Configure the Streamlit page
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

# 2. FastAPI prediction endpoint
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict",
)

# 3. Page heading
st.title("Customer Churn Prediction")
st.write(
    "Analyze customer information to estimate the likelihood "
    "of a customer leaving the service."
)

st.caption(
    "This application uses a trained Gradient Boosting model "
    "through a FastAPI backend."
)

st.divider()


# 4. Customer input form
with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"],
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"],
        )

    with col2:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0,
        )

    st.divider()
    st.subheader("Additional Services")

    service_col1, service_col2, service_col3 = st.columns(3)

    with service_col1:
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
        )

    with service_col2:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
        )

    with service_col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
        )

    st.divider()
    st.subheader("Billing Information")

    billing_col1, billing_col2 = st.columns(2)

    with billing_col1:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0,
            help="Enter the customer's accumulated charges.",
        )

    with billing_col2:
        st.caption(
            "Tip: Total Charges generally reflects the "
            "customer's accumulated charges. Keep the value "
            "consistent with tenure and monthly charges."
        )

    predict_button = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True,
    )


# 5. Send the customer details to FastAPI
if predict_button:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        with st.spinner("Analyzing customer details..."):
            response = requests.post(
                API_URL,
                json=customer_data,
                timeout=15,
            )

        if response.ok:
            result = response.json()

            prediction = result["predicted_churn"]
            probability = float(result["churn_probability"])
            threshold = result.get("threshold")
            model_name = result.get("model", "ML model")

            st.divider()
            st.subheader("Prediction Results")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Predicted Churn",
                    prediction,
                )

            with metric2:
                st.metric(
                    "Churn Probability",
                    f"{probability:.2%}",
                )

            if prediction == "Yes":
                st.warning(
                    "Churn predicted: this customer may be at "
                    "risk of leaving the service."
                )
            else:
                st.success(
                    "Churn not predicted for this customer "
                    "at the model's current decision threshold."
                )

            st.write("**Churn probability indicator**")
            st.progress(
                max(0.0, min(1.0, probability))
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.write(f"**Model:** {model_name}")

            with result_col2:
                if threshold is not None:
                    st.write(
                        f"**Decision threshold:** {float(threshold):.2%}"
                    )

            st.info(
                "The probability is a model estimate, not a "
                "guarantee of future customer behavior."
            )

        else:
            st.error(
                f"API error ({response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the FastAPI backend. "
            "Make sure the API is running and accessible "
            f"at {API_URL}."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out. Please try again."
        )

    except (ValueError, KeyError, TypeError):
        st.error(
            "The API response was missing expected fields "
            "or contained invalid data. Check the backend response."
        )
