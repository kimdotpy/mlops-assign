"""Drift monitoring with Evidently AI (Evidently 0.7+ compatible)."""
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.metrics import (
    DriftedColumnsCount,
    ValueDrift,
)
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    DRIFT_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    DRIFT_THRESHOLDS,
)
from src.monitor.custom_metrics import (
    calculate_mean_monthly_charges_diff,
    calculate_churn_rate_by_contract_shift,
    calculate_mean_tenure_diff,
)


def load_drift_data():
    """Load reference and current (drifted) datasets."""
    ref_path = DRIFT_DATA_DIR / "reference.csv"
    cur_path = DRIFT_DATA_DIR / "current_drifted.csv"
    
    if not ref_path.exists() or not cur_path.exists():
        raise FileNotFoundError(
            f"Drift data not found. Run data preparation first. "
            f"Expected: {ref_path}, {cur_path}"
        )
    
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)
    
    return ref_df, cur_df


def create_data_drift_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame):
    """Create comprehensive data drift report."""
    
    metrics = [
        DataDriftPreset(),
        DriftedColumnsCount(),
        ValueDrift(column="MonthlyCharges"),
        ValueDrift(column="tenure"),
        ValueDrift(column="TotalCharges"),
    ]
    
    contract_cols = [c for c in ref_df.columns if c.startswith("Contract_")]
    for col in contract_cols:
        metrics.append(ValueDrift(column=col))
    
    report = Report(metrics=metrics)
    snapshot = report.run(reference_data=ref_df, current_data=cur_df)
    
    return report, snapshot


def create_target_drift_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame):
    """Create target drift report."""
    
    if TARGET_COLUMN not in ref_df.columns or TARGET_COLUMN not in cur_df.columns:
        print(f"Target column {TARGET_COLUMN} not found in data")
        return None, None
    
    report = Report(metrics=[
        ValueDrift(column=TARGET_COLUMN),
    ])
    
    snapshot = report.run(reference_data=ref_df, current_data=cur_df)
    return report, snapshot


def analyze_drift_results(data_drift_snapshot, target_drift_snapshot, 
                          ref_df: pd.DataFrame, cur_df: pd.DataFrame):
    """Analyze and print drift detection results."""
    print("\n" + "="*60)
    print("DRIFT DETECTION RESULTS")
    print("="*60)
    
    data_drift_result = data_drift_snapshot.dict()
    
    # Check dataset-level drift from DataDriftPreset
    dataset_drift = False
    for metric in data_drift_result.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        if "DataDrift" in metric_name:
            # The DataDriftPreset result is in the value field
            value = metric.get("value", {})
            if isinstance(value, dict):
                dataset_drift = value.get("dataset_drift", False)
            break
    
    print(f"\nDataset Drift Detected: {dataset_drift}")
    
    print("\n--- Column-Level Drift ---")
    for metric in data_drift_result.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        value = metric.get("value", {})
        
        if "ValueDrift" in metric_name:
            column = metric.get("config", {}).get("column", "unknown")
            drift_detected = value.get("drift_detected", False) if isinstance(value, dict) else False
            drift_score = value.get("drift_score", None) if isinstance(value, dict) else None
            stattest_name = value.get("stattest_name", "unknown") if isinstance(value, dict) else "unknown"
            
            status = "DRIFT" if drift_detected else "OK"
            print(f"  {column}: {status} (drift_score: {drift_score}, test: {stattest_name})")
    
    # Custom metric calculations
    print("\n--- Custom Drift Metrics ---")
    
    # Mean MonthlyCharges Diff
    mcd = calculate_mean_monthly_charges_diff(ref_df, cur_df, "MonthlyCharges")
    value = mcd["value"]
    ref_mean = mcd["reference_mean"]
    cur_mean = mcd["current_mean"]
    threshold = DRIFT_THRESHOLDS["monthly_charges_mean_diff_threshold"]
    status = "EXCEEDS THRESHOLD" if value > threshold else "OK"
    print(f"  Mean MonthlyCharges Diff: {value:.2f} (ref: {ref_mean:.2f}, cur: {cur_mean:.2f}) - {status}")
    
    # Mean Tenure Diff
    mtd = calculate_mean_tenure_diff(ref_df, cur_df, "tenure")
    value = mtd["value"]
    ref_mean = mtd["reference_mean"]
    cur_mean = mtd["current_mean"]
    print(f"  Mean Tenure Diff: {value:.2f} (ref: {ref_mean:.2f}, cur: {cur_mean:.2f})")
    
    # Churn Rate by Contract Shift
    ccs = calculate_churn_rate_by_contract_shift(ref_df, cur_df)
    value = ccs["value"]
    ref_rate = ccs["reference_churn_rate"]
    cur_rate = ccs["current_churn_rate"]
    threshold = DRIFT_THRESHOLDS["churn_rate_diff_threshold"]
    status = "EXCEEDS THRESHOLD" if value > threshold else "OK"
    print(f"  Churn Rate Shift (Month-to-month): {value:.4f} (ref: {ref_rate:.4f}, cur: {cur_rate:.4f}) - {status}")
    
    if target_drift_snapshot:
        print("\n--- Target Drift ---")
        target_result = target_drift_snapshot.dict()
        for metric in target_result.get("metrics", []):
            metric_name = metric.get("metric_name", "")
            value = metric.get("value", {})
            
            if "ValueDrift" in metric_name:
                column = metric.get("config", {}).get("column", "unknown")
                drift_detected = value.get("drift_detected", False) if isinstance(value, dict) else False
                drift_score = value.get("drift_score", None) if isinstance(value, dict) else None
                print(f"  {column}: {'DRIFT' if drift_detected else 'OK'} (drift_score: {drift_score})")


def log_reports_to_mlflow(data_drift_snapshot, target_drift_snapshot,
                          ref_df: pd.DataFrame, cur_df: pd.DataFrame):
    """Save reports as HTML and log to MLflow."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    data_drift_path = REPORTS_DIR / "data_drift_report.html"
    data_drift_snapshot.save_html(str(data_drift_path))
    print(f"\nData drift report saved to: {data_drift_path}")
    
    target_drift_path = None
    if target_drift_snapshot:
        target_drift_path = REPORTS_DIR / "target_drift_report.html"
        target_drift_snapshot.save_html(str(target_drift_path))
        print(f"Target drift report saved to: {target_drift_path}")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="drift_monitoring", nested=True):
        mlflow.log_artifact(str(data_drift_path), "drift_reports")
        if target_drift_path:
            mlflow.log_artifact(str(target_drift_path), "drift_reports")
        
        # Log custom metrics
        mcd = calculate_mean_monthly_charges_diff(ref_df, cur_df, "MonthlyCharges")
        mlflow.log_metric("drift_monthly_charges_mean_diff", mcd["value"])
        
        mtd = calculate_mean_tenure_diff(ref_df, cur_df, "tenure")
        mlflow.log_metric("drift_tenure_mean_diff", mtd["value"])
        
        ccs = calculate_churn_rate_by_contract_shift(ref_df, cur_df)
        mlflow.log_metric("drift_churn_rate_contract_shift", ccs["value"])
        
        # Log column-level drift metrics
        data_drift_result = data_drift_snapshot.dict()
        for metric in data_drift_result.get("metrics", []):
            metric_name = metric.get("metric_name", "")
            value = metric.get("value", {})
            
            if "ValueDrift" in metric_name:
                column = metric.get("config", {}).get("column", "unknown")
                # Sanitize column name for MLflow metric naming
                safe_column = column.replace("(", "").replace(")", "").replace(" ", "_").replace("-", "_")
                drift_score = value.get("drift_score", 1.0) if isinstance(value, dict) else 1.0
                drift_detected = value.get("drift_detected", False) if isinstance(value, dict) else False
                mlflow.log_metric(f"drift_score_{safe_column}", drift_score)
                mlflow.log_metric(f"drift_detected_{safe_column}", int(drift_detected))


def run_drift_check():
    """Main drift check pipeline."""
    print("Starting drift monitoring check...")
    
    ref_df, cur_df = load_drift_data()
    print(f"Reference data: {len(ref_df)} rows, {len(ref_df.columns)} columns")
    print(f"Current data: {len(cur_df)} rows, {len(cur_df.columns)} columns")
    
    print("\nGenerating data drift report...")
    _, data_drift_snapshot = create_data_drift_report(ref_df, cur_df)
    
    print("Generating target drift report...")
    _, target_drift_snapshot = create_target_drift_report(ref_df, cur_df)
    
    analyze_drift_results(data_drift_snapshot, target_drift_snapshot, ref_df, cur_df)
    
    print("\nLogging reports to MLflow...")
    log_reports_to_mlflow(data_drift_snapshot, target_drift_snapshot, ref_df, cur_df)
    
    print("\nDrift check complete!")
    
    return data_drift_snapshot, target_drift_snapshot


def main():
    """Entry point for drift check."""
    run_drift_check()


if __name__ == "__main__":
    main()