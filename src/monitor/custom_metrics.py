"""Utility functions for drift monitoring (Evidently 0.7+ compatible).
These replace custom Metric classes which are more complex in Evidently 0.7+."""
import pandas as pd
import numpy as np


def calculate_mean_monthly_charges_diff(ref_df: pd.DataFrame, cur_df: pd.DataFrame, column_name: str = "MonthlyCharges") -> dict:
    """Calculate absolute difference in mean MonthlyCharges between reference and current."""
    if column_name not in ref_df.columns or column_name not in cur_df.columns:
        raise ValueError(f"Column {column_name} not found in data")
    
    ref_mean = ref_df[column_name].mean()
    cur_mean = cur_df[column_name].mean()
    diff = abs(cur_mean - ref_mean)
    
    return {
        "value": diff,
        "column_name": column_name,
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
    }


def calculate_churn_rate_by_contract_shift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, 
                                           contract_column: str = "Contract_Month-to-month", 
                                           target_column: str = "Churn") -> dict:
    """Calculate shift in churn rate for Month-to-month contract segment.
    
    Month-to-month is the reference category (when Contract_One year=0 and Contract_Two year=0).
    """
    # Find Month-to-month: both Contract_One year and Contract_Two year are 0
    ref_mtm = ref_df[(ref_df.get("Contract_One year", 0) == 0) & (ref_df.get("Contract_Two year", 0) == 0)]
    cur_mtm = cur_df[(cur_df.get("Contract_One year", 0) == 0) & (cur_df.get("Contract_Two year", 0) == 0)]
    
    if len(ref_mtm) == 0 or len(cur_mtm) == 0:
        return {
            "value": 0.0,
            "reference_churn_rate": 0.0,
            "current_churn_rate": 0.0,
            "reference_count": len(ref_mtm),
            "current_count": len(cur_mtm),
        }
    
    ref_churn_rate = ref_mtm[target_column].mean()
    cur_churn_rate = cur_mtm[target_column].mean()
    diff = abs(cur_churn_rate - ref_churn_rate)
    
    return {
        "value": diff,
        "reference_churn_rate": ref_churn_rate,
        "current_churn_rate": cur_churn_rate,
        "reference_count": len(ref_mtm),
        "current_count": len(cur_mtm),
    }


def calculate_mean_tenure_diff(ref_df: pd.DataFrame, cur_df: pd.DataFrame, column_name: str = "tenure") -> dict:
    """Calculate absolute difference in mean tenure between reference and current."""
    if column_name not in ref_df.columns or column_name not in cur_df.columns:
        raise ValueError(f"Column {column_name} not found in data")
    
    ref_mean = ref_df[column_name].mean()
    cur_mean = cur_df[column_name].mean()
    diff = abs(cur_mean - ref_mean)
    
    return {
        "value": diff,
        "column_name": column_name,
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
    }