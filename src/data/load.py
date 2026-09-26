"""Data loading and preprocessing utilities."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_raw_data(filepath: Path = None) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw data."""
    df = df.copy()
    
    # Handle TotalCharges - convert to numeric, coerce errors to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Fill missing TotalCharges with MonthlyCharges * tenure (for new customers)
    missing_mask = df["TotalCharges"].isna()
    df.loc[missing_mask, "TotalCharges"] = df.loc[missing_mask, "MonthlyCharges"] * df.loc[missing_mask, "tenure"]
    
    # Drop customerID as it's not a predictive feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    
    # Convert target to binary (Yes/No -> 1/0)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    
    # Identify categorical and numeric columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(TARGET_COLUMN)  # Remove target from features
    
    return df, categorical_cols, numeric_cols


def encode_features(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """One-hot encode categorical features."""
    df = df.copy()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def prepare_features_target(df: pd.DataFrame) -> tuple:
    """Split dataframe into features X and target y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> tuple:
    """Stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_processed_data(X_train, X_test, y_train, y_test, prefix: str = "reference"):
    """Save processed train/test splits."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(PROCESSED_DATA_DIR / f"{prefix}_train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / f"{prefix}_test.csv", index=False)
    
    return train_df, test_df


def load_processed_data(prefix: str = "reference") -> tuple:
    """Load processed train/test splits."""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / f"{prefix}_train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / f"{prefix}_test.csv")
    
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    
    return X_train, X_test, y_train, y_test