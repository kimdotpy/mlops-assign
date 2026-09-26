"""Unit tests for custom drift metrics."""
import pandas as pd
import numpy as np
from src.monitor.custom_metrics import (
    calculate_mean_monthly_charges_diff,
    calculate_churn_rate_by_contract_shift,
    calculate_mean_tenure_diff,
)


def test_calculate_mean_monthly_charges_diff():
    """Test mean MonthlyCharges difference calculation."""
    ref_df = pd.DataFrame({"MonthlyCharges": [50.0, 60.0, 70.0, 80.0]})
    cur_df = pd.DataFrame({"MonthlyCharges": [60.0, 70.0, 80.0, 90.0]})

    result = calculate_mean_monthly_charges_diff(ref_df, cur_df, "MonthlyCharges")

    assert "value" in result
    assert "reference_mean" in result
    assert "current_mean" in result
    assert result["reference_mean"] == 65.0
    assert result["current_mean"] == 75.0
    assert result["value"] == 10.0  # |75 - 65| = 10


def test_calculate_mean_monthly_charges_diff_missing_column():
    """Test error when column is missing."""
    ref_df = pd.DataFrame({"MonthlyCharges": [50.0, 60.0]})
    cur_df = pd.DataFrame({"OtherCol": [60.0, 70.0]})

    try:
        calculate_mean_monthly_charges_diff(ref_df, cur_df, "MonthlyCharges")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "MonthlyCharges" in str(e)


def test_calculate_churn_rate_by_contract_shift():
    """Test churn rate shift for Month-to-month contracts."""
    # Month-to-month = Contract_One_year=0 AND Contract_Two_year=0
    ref_df = pd.DataFrame({
        "Contract_One year": [0, 0, 1, 0],
        "Contract_Two year": [0, 0, 0, 1],
        "Churn": [1, 0, 1, 0],  # 2 Month-to-month, 1 churn = 50%
    })
    cur_df = pd.DataFrame({
        "Contract_One year": [0, 0, 0, 1],
        "Contract_Two year": [0, 0, 0, 0],
        "Churn": [1, 1, 0, 0],  # 3 Month-to-month, 2 churn = 66.7%
    })

    result = calculate_churn_rate_by_contract_shift(ref_df, cur_df)

    assert "value" in result
    assert "reference_churn_rate" in result
    assert "current_churn_rate" in result
    assert "reference_count" in result
    assert "current_count" in result
    assert result["reference_count"] == 2
    assert result["current_count"] == 3
    assert abs(result["reference_churn_rate"] - 0.5) < 0.01
    assert abs(result["current_churn_rate"] - 2/3) < 0.01
    assert abs(result["value"] - abs(2/3 - 0.5)) < 0.01


def test_calculate_churn_rate_by_contract_shift_no_mtm():
    """Test when no Month-to-month contracts exist."""
    ref_df = pd.DataFrame({
        "Contract_One year": [1, 1],
        "Contract_Two year": [0, 0],
        "Churn": [1, 0],
    })
    cur_df = pd.DataFrame({
        "Contract_One year": [1, 1],
        "Contract_Two year": [0, 0],
        "Churn": [0, 1],
    })

    result = calculate_churn_rate_by_contract_shift(ref_df, cur_df)

    assert result["value"] == 0.0
    assert result["reference_churn_rate"] == 0.0
    assert result["current_churn_rate"] == 0.0
    assert result["reference_count"] == 0
    assert result["current_count"] == 0


def test_calculate_mean_tenure_diff():
    """Test mean tenure difference calculation."""
    ref_df = pd.DataFrame({"tenure": [10, 20, 30, 40]})
    cur_df = pd.DataFrame({"tenure": [15, 25, 35, 45]})

    result = calculate_mean_tenure_diff(ref_df, cur_df, "tenure")

    assert "value" in result
    assert "reference_mean" in result
    assert "current_mean" in result
    assert result["reference_mean"] == 25.0
    assert result["current_mean"] == 30.0
    assert result["value"] == 5.0  # |30 - 25| = 5


def test_calculate_mean_tenure_diff_missing_column():
    """Test error when tenure column is missing."""
    ref_df = pd.DataFrame({"tenure": [10, 20]})
    cur_df = pd.DataFrame({"OtherCol": [15, 25]})

    try:
        calculate_mean_tenure_diff(ref_df, cur_df, "tenure")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "tenure" in str(e)