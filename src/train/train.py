"""Main training script with MLflow experiment tracking."""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from src.data.load import load_processed_data, prepare_features_target
from src.train.models import get_all_models
from src.train.evaluate import compute_metrics, evaluate_model


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_model(model, X_train, y_train):
    """Train a single model."""
    model.fit(X_train, y_train)
    return model


def log_model_artifacts(model, model_name, X_test, y_test, metrics, run_id):
    """Log model, metrics, and plots to MLflow."""
    # Log parameters
    mlflow.log_params(model.get_params())
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log model with signature
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_test, model.predict(X_test))
    
    # Determine trusted types based on model type
    model_type = type(model).__name__
    if "XGB" in model_type or "XGBoost" in model_type:
        trusted_types = ["sklearn.tree._tree.Tree", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
    elif "RandomForest" in model_type or "DecisionTree" in model_type:
        trusted_types = ["sklearn.tree._tree.Tree"]
    else:
        trusted_types = None
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.iloc[:5],
        registered_model_name=MLFLOW_MODEL_NAME,
        skops_trusted_types=trusted_types,
    )
    
    # Log confusion matrix and ROC curve as artifacts
    # These were saved by evaluate_model, we just need to log them
    artifacts_dir = Path("artifacts") / model_name
    if artifacts_dir.exists():
        for artifact in artifacts_dir.glob("*"):
            mlflow.log_artifact(str(artifact))


def run_training():
    """Main training pipeline."""
    setup_mlflow()
    
    # Load processed reference data
    print("Loading reference training data...")
    X_train, X_test, y_train, y_test = load_processed_data(prefix="reference")
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Churn rate in train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")
    
    # Get all model configurations
    models = get_all_models()
    
    best_model = None
    best_f1 = 0
    best_run_id = None
    best_model_name = None
    
    for model_config in models:
        model_name = model_config["name"]
        model = model_config["model"]
        
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            
            # Log model type and hyperparameters
            mlflow.log_param("model_type", model_config["model_type"])
            mlflow.log_param("model_name", model_name)
            
            # Train model
            model = train_model(model, X_train, y_train)
            
            # Evaluate
            metrics, y_pred, y_pred_proba = evaluate_model(
                model, X_test, y_test, model_name, save_dir=Path("artifacts") / model_name
            )
            
            # Print metrics
            print(f"Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Log to MLflow
            log_model_artifacts(model, model_name, X_test, y_test, metrics, run_id)
            
            # Track best model (by F1 score)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_run_id = run_id
                best_model_name = model_name
    
    print(f"\n{'='*60}")
    print(f"Best model: {best_model_name} (F1: {best_f1:.4f})")
    print(f"Best run ID: {best_run_id}")
    print(f"{'='*60}")
    
    return best_model, best_model_name, best_run_id, best_f1


def register_best_model(run_id: int, model_name: str):
    """Register the best model in MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Registering model from {model_uri}...")
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MLFLOW_MODEL_NAME,
    )
    
    print(f"Model registered: {registered_model.name} version {registered_model.version}")
    return registered_model


def transition_model_stage(model_name: str, version: int, stage: str):
    """Transition model to a specific stage."""
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    print(f"Model {model_name} v{version} transitioned to {stage}")


def main():
    """Main entry point."""
    print("Starting Telco Churn Model Training with MLflow Tracking")
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    # Run training
    best_model, best_model_name, best_run_id, best_f1 = run_training()
    
    # Register best model
    registered_model = register_best_model(best_run_id, best_model_name)
    
    # Transition through stages: Staging -> Production
    print("\nTransitioning model through stages...")
    transition_model_stage(MLFLOW_MODEL_NAME, registered_model.version, "Staging")
    transition_model_stage(MLFLOW_MODEL_NAME, registered_model.version, "Production")
    
    print("\nTraining complete!")
    print(f"Best model: {best_model_name}")
    print(f"F1 Score: {best_f1:.4f}")
    print(f"Registered as: {MLFLOW_MODEL_NAME} v{registered_model.version} (Production)")
    
    return best_model, registered_model


if __name__ == "__main__":
    main()