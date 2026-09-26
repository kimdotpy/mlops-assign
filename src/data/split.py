"""Data splitting and synthetic drift injection."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DRIFT_DATA_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    DRIFT_CONFIG,
)
from src.data.load import (
    load_raw_data, clean_data, encode_features, 
    prepare_features_target, save_processed_data, split_data
)


def create_reference_current_split(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE) -> tuple:
    """Create reference (70%) and current (30%) datasets from raw data."""
    df = load_raw_data()
    df, categorical_cols, numeric_cols = clean_data(df)
    df = encode_features(df, categorical_cols)
    
    ref_df, cur_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COLUMN]
    )
    return ref_df, cur_df


def inject_drift(current_df: pd.DataFrame, drift_config: dict = None) -> pd.DataFrame:
    """Inject synthetic drift into the current dataset."""
    if drift_config is None:
        drift_config = DRIFT_CONFIG
    
    df = current_df.copy()
    np.random.seed(RANDOM_STATE)
    
    # 1. Numeric drift: Shift MonthlyCharges
    if "MonthlyCharges" in df.columns:
        shift = np.random.normal(
            drift_config["monthly_charges_shift_mean"],
            drift_config["monthly_charges_shift_std"],
            size=len(df)
        )
        df["MonthlyCharges"] = df["MonthlyCharges"] + shift
        df["MonthlyCharges"] = df["MonthlyCharges"].clip(lower=0)
    
    # 2. Categorical drift: Oversample Contract == "Month-to-month"
    contract_cols = [c for c in df.columns if c.startswith("Contract_")]
    month_to_month_col = None
    for col in contract_cols:
        if "Month-to-month" in col or "Month_to_month" in col:
            month_to_month_col = col
            break
    
    if month_to_month_col is not None and month_to_month_col in df.columns:
        target_ratio = drift_config["contract_month_to_month_target_ratio"]
        current_ratio = df[month_to_month_col].mean()
        
        if target_ratio > current_ratio:
            mtm_indices = df[df[month_to_month_col] == 1].index
            other_indices = df[df[month_to_month_col] == 0].index
            
            n_current_mtm = len(mtm_indices)
            n_total = len(df)
            n_needed = int(target_ratio * n_total) - n_current_mtm
            
            if n_needed > 0:
                additional_indices = np.random.choice(mtm_indices, size=n_needed, replace=True)
                additional_rows = df.loc[additional_indices].copy()
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != TARGET_COLUMN:
                        noise = np.random.normal(0, df[col].std() * 0.01, size=len(additional_rows))
                        additional_rows[col] = additional_rows[col] + noise
                
                df = pd.concat([df, additional_rows], ignore_index=True)
    
    # 3. Label drift: Flip a small percentage of target labels
    if drift_config.get("label_flip_rate", 0) > 0:
        flip_mask = np.random.random(len(df)) < drift_config["label_flip_rate"]
        df.loc[flip_mask, TARGET_COLUMN] = 1 - df.loc[flip_mask, TARGET_COLUMN]
    
    return df


def prepare_datasets():
    """Main function to prepare reference and current (drift-injected) datasets."""
    print("Loading and cleaning raw data...")
    ref_df, cur_df = create_reference_current_split()
    
    print("Injecting synthetic drift into current dataset...")
    cur_df_drifted = inject_drift(cur_df)
    
    # Reference split
    X_ref, y_ref = prepare_features_target(ref_df)
    X_ref_train, X_ref_test, y_ref_train, y_ref_test = split_data(X_ref, y_ref)
    save_processed_data(X_ref_train, X_ref_test, y_ref_train, y_ref_test, prefix="reference")
    
    # Current (drifted) split
    X_cur, y_cur = prepare_features_target(cur_df_drifted)
    X_cur_train, X_cur_test, y_cur_train, y_cur_test = split_data(X_cur, y_cur)
    save_processed_data(X_cur_train, X_cur_test, y_cur_train, y_cur_test, prefix="current")
    
    # Save full drifted datasets for drift monitoring
    DRIFT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cur_df_drifted.to_csv(DRIFT_DATA_DIR / "current_drifted.csv", index=False)
    ref_df.to_csv(DRIFT_DATA_DIR / "reference.csv", index=False)
    
    print(f"Reference dataset: {len(ref_df)} samples")
    print(f"Current dataset (drifted): {len(cur_df_drifted)} samples")
    print(f"Reference train: {len(X_ref_train)}, test: {len(X_ref_test)}")
    print(f"Current train: {len(X_cur_train)}, test: {len(X_cur_test)}")
    
    # Print drift summary
    print("\n--- Drift Summary ---")
    print(f"Reference MonthlyCharges mean: {ref_df['MonthlyCharges'].mean():.2f}")
    print(f"Current MonthlyCharges mean: {cur_df_drifted['MonthlyCharges'].mean():.2f}")
    print(f"Reference Churn rate: {ref_df[TARGET_COLUMN].mean():.4f}")
    print(f"Current Churn rate: {cur_df_drifted[TARGET_COLUMN].mean():.4f}")
    
    contract_cols = [c for c in ref_df.columns if c.startswith("Contract_")]
    for col in contract_cols:
        ref_ratio = ref_df[col].mean()
        cur_ratio = cur_df_drifted[col].mean()
        print(f"Reference {col}: {ref_ratio:.4f}, Current: {cur_ratio:.4f}")
    
    return ref_df, cur_df_drifted


if __name__ == "__main__":
    prepare_datasets()