"""Model evaluation utilities: metrics computation and plotting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from src.config import REPORTS_DIR


def compute_metrics(y_true, y_pred, y_pred_proba):
    """Compute all classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None, title: str = "Confusion Matrix"):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return cm


def plot_roc_curve(y_true, y_pred_proba, save_path: Path = None, title: str = "ROC Curve"):
    """Plot and optionally save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return fpr, tpr, auc


def plot_feature_importance(model, feature_names, save_path: Path = None, title: str = "Feature Importance", top_n: int = 20):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def evaluate_model(model, X_test, y_test, model_name: str, save_dir: Path = None):
    """Full model evaluation with metrics and plots."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        cm_path = save_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, save_path=cm_path, title=f"{model_name} - Confusion Matrix")
        
        # ROC curve
        roc_path = save_dir / f"{model_name}_roc_curve.png"
        plot_roc_curve(y_test, y_pred_proba, save_path=roc_path, title=f"{model_name} - ROC Curve")
        
        # Feature importance (if available)
        fi_path = save_dir / f"{model_name}_feature_importance.png"
        plot_feature_importance(model, X_test.columns.tolist(), save_path=fi_path, title=f"{model_name} - Feature Importance")
    
    return metrics, y_pred, y_pred_proba


def print_classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]):
    """Print detailed classification report."""
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    return report