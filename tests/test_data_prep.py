"""Unit tests for data preparation and drift injection."""
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
from src.data.load import (
    load_raw_data,
    clean_data,
    encode_features,
    prepare_features_target,
    split_data,
    save_processed_data,
)
from src.data.split import (
    inject_drift,
    create_reference_current_split,
    prepare_datasets,
)


def test_load_raw_data():
    """Test loading raw Telco data."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Churn" in df.columns
    assert "customerID" in df.columns


def test_clean_data():
    """Test data cleaning converts target and handles TotalCharges."""
    # Create minimal test data
    df = pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 24, 6],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
        "MonthlyCharges": [70.0, 80.0, 90.0],
        "TotalCharges": ["1000.0", "2000.0", " "],  # Last one is invalid
        "Churn": ["Yes", "No", "Yes"],
    })

    cleaned_df, cat_cols, num_cols = clean_data(df)

    # Check customerID dropped
    assert "customerID" not in cleaned_df.columns

    # Check Churn is binary
    assert set(cleaned_df["Churn"].unique()).issubset({0, 1})

    # Check TotalCharges is numeric
    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])

    # Check categorical and numeric columns identified
    assert len(cat_cols) > 0
    assert len(num_cols) > 0
    assert "Churn" not in num_cols


def test_encode_features():
    """Test one-hot encoding of categorical features."""
    df = pd.DataFrame({
        "gender": ["Male", "Female", "Male"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "tenure": [12, 24, 6],
        "Churn": [1, 0, 1],
    })

    cat_cols = ["gender", "Contract"]
    encoded = encode_features(df, cat_cols)

    assert "gender_Male" in encoded.columns
    # With drop_first=True, first alphabetical category is dropped
    # "Month-to-month" < "One year" < "Two year" alphabetically
    # So "Month-to-month" is dropped, leaving "One year" and "Two year"
    assert "Contract_One year" in encoded.columns
    assert "Contract_Two year" in encoded.columns
    assert "gender" not in encoded.columns
    assert "Contract" not in encoded.columns


def test_inject_drift():
    """Test drift injection creates expected shifts."""
    np.random.seed(42)

    # Create reference data
    ref_df = pd.DataFrame({
        "MonthlyCharges": [50.0] * 100 + [100.0] * 100,  # mean = 75
        "tenure": [20] * 200,  # mean = 20
        "Contract_Month-to-month": [1] * 100 + [0] * 100,  # 50% Month-to-month
        "Contract_One year": [0] * 100 + [1] * 100,
        "Contract_Two year": [0] * 200,
        "Churn": [0] * 150 + [1] * 50,  # 25% churn
    })

    cur_df = inject_drift(ref_df.copy())

    # Check MonthlyCharges shifted up by ~$10
    assert cur_df["MonthlyCharges"].mean() > ref_df["MonthlyCharges"].mean()
    diff = cur_df["MonthlyCharges"].mean() - ref_df["MonthlyCharges"].mean()
    assert 8 < diff < 12  # approximately $10 shift

    # Check Month-to-month oversampled (should be > 50%)
    mtm_ratio = cur_df["Contract_Month-to-month"].mean()
    assert mtm_ratio > 0.6  # oversampled

    # Check label flip (approximately 5% of rows flipped)
    churn_diff = abs(cur_df["Churn"].mean() - ref_df["Churn"].mean())
    assert 0.03 < churn_diff < 0.08  # approximately 5% flip


def test_create_reference_current_split():
    """Test reference/current split maintains proportions."""
    ref_df, cur_df = create_reference_current_split(test_size=0.3, random_state=42)

    assert len(ref_df) > 0
    assert len(cur_df) > 0
    # Total should be 70/30 split
    total = len(ref_df) + len(cur_df)
    assert abs(len(ref_df) / total - 0.7) < 0.05
    assert abs(len(cur_df) / total - 0.3) < 0.05

    # Churn distribution should be similar (stratified)
    ref_churn_rate = ref_df["Churn"].mean()
    cur_churn_rate = cur_df["Churn"].mean()
    assert abs(ref_churn_rate - cur_churn_rate) < 0.05


def test_save_processed_data():
    """Test saving processed train/test splits."""
    X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    X_test = pd.DataFrame({"a": [5], "b": [6]})
    y_train = pd.Series([0, 1], name="Churn")
    y_test = pd.Series([1], name="Churn")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use the function's internal PROCESSED_DATA_DIR by mocking it
        import src.data.load as load_module
        from pathlib import Path
        original_dir = load_module.PROCESSED_DATA_DIR
        load_module.PROCESSED_DATA_DIR = Path(tmpdir)

        try:
            save_processed_data(X_train, X_test, y_train, y_test, prefix="test")
            assert os.path.exists(os.path.join(tmpdir, "test_train.csv"))
            assert os.path.exists(os.path.join(tmpdir, "test_test.csv"))
        finally:
            load_module.PROCESSED_DATA_DIR = original_dir