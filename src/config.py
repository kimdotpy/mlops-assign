"""Central configuration for Telco Churn MLOps pipeline."""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"  # Allow file-based tracking for MLflow 3.x
from pathlib import Path

# Set matplotlib backend to avoid tkinter issues
import matplotlib
matplotlib.use("Agg")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DRIFT_DATA_DIR = DATA_DIR / "drift_injected"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# MLflow Configuration
# Use local path for Windows compatibility
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(MLRUNS_DIR))
MLFLOW_EXPERIMENT_NAME = "telco-churn"
MLFLOW_MODEL_NAME = "telco-churn-model"
MLFLOW_REGISTRY_STAGES = ["Staging", "Production"]

# Data Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.3
TARGET_COLUMN = "Churn"

# Model Training Configuration
MODEL_CONFIGS = [
    {
        "name": "RandomForest_Shallow",
        "model_type": "RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    {
        "name": "RandomForest_Deep",
        "model_type": "RandomForestClassifier",
        "params": {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    {
        "name": "LogisticRegression_L2",
        "model_type": "LogisticRegression",
        "params": {
            "C": 0.1,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    },
    {
        "name": "XGBoost",
        "model_type": "XGBClassifier",
        "params": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 3,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "logloss",
        },
    },
]

# Drift Injection Configuration
DRIFT_CONFIG = {
    "monthly_charges_shift_mean": 15.0,
    "monthly_charges_shift_std": 5.0,
    "contract_month_to_month_target_ratio": 0.75,  # oversample to 75%
    "label_flip_rate": 0.05,
}

# Drift Detection Thresholds
DRIFT_THRESHOLDS = {
    "p_value_threshold": 0.05,
    "monthly_charges_mean_diff_threshold": 10.0,
    "churn_rate_diff_threshold": 0.1,
}

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_URI = f"models:/{MLFLOW_MODEL_NAME}/Production"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DRIFT_DATA_DIR, REPORTS_DIR, MLRUNS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)