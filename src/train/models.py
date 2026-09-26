"""Model definitions and hyperparameter configurations."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.config import MODEL_CONFIGS


def get_model_configs():
    """Return list of model configurations for training."""
    return MODEL_CONFIGS


def create_model(model_type: str, params: dict):
    """Instantiate a model based on type and parameters."""
    model_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "XGBClassifier": XGBClassifier,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](**params)


def get_all_models():
    """Create all models from configs."""
    models = []
    for config in MODEL_CONFIGS:
        model = create_model(config["model_type"], config["params"])
        models.append({
            "name": config["name"],
            "model": model,
            "params": config["params"],
            "model_type": config["model_type"],
        })
    return models