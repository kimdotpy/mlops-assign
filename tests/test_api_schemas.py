"""Unit tests for API Pydantic schemas."""
from src.serve.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)


def test_customer_features_valid():
    """Test valid CustomerFeatures input."""
    data = {
        "tenure": 12,
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.2,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    customer = CustomerFeatures(**data)
    assert customer.tenure == 12
    assert customer.MonthlyCharges == 70.35
    assert customer.gender.value == "Male"


def test_customer_features_invalid_tenure():
    """Test tenure validation (must be >= 0)."""
    data = {
        "tenure": -1,
        "MonthlyCharges": 70.0,
        "TotalCharges": 844.2,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    try:
        CustomerFeatures(**data)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "tenure" in str(e).lower() or "ge=0" in str(e)


def test_customer_features_invalid_binary():
    """Test binary fields must be 0 or 1."""
    data = {
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 844.2,
        "gender": "Male",
        "SeniorCitizen": 2,  # Invalid: must be 0 or 1
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    try:
        CustomerFeatures(**data)
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "SeniorCitizen" in str(e) or "le=1" in str(e)