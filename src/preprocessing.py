import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the tabular dataset from a specified path."""
    if not filepath:
        raise ValueError("Filepath cannot be empty or None")
    return pd.read_csv(filepath)

def handle_missing_values(df: pd.DataFrame, numeric_cols: list, strategy: str = "median") -> pd.DataFrame:
    """Imputes missing values in numeric columns without altering the original DataFrame."""
    df_clean = df.copy()
    for col in numeric_cols:
        if col in df_clean.columns:
            if df_clean[col].dtype == object:
                df_clean[col] = pd.to_numeric(df_clean[col].replace('?', np.nan))
            
            if strategy == "median":
                fill_value = df_clean[col].median()
            elif strategy == "mean":
                fill_value = df_clean[col].mean()
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}")
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    return df_clean

def encode_categories(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Encodes categorical columns to numeric codes via pandas categorical features."""
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype("category").cat.codes
    return df_encoded

def process_and_split(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    """Splits target data from features into train and test evaluation splits."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def process_data():
    """Master workflow to run preprocessing steps and save train/test files."""
    # Load configuration parameters
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    d_cfg = config["data"]
    p_cfg = config["preprocessing"]
    
    # Run data pipeline stages
    df_raw = load_data(d_cfg["raw_path"])
    df_imputed = handle_missing_values(df_raw, p_cfg["numeric_cols"], p_cfg["impute_strategy"])
    df_final = encode_categories(df_imputed, p_cfg["categorical_cols"])
    
    X_train, X_test, y_train, y_test = process_and_split(
        df_final, 
        target_col=d_cfg["target_col"], 
        test_size=d_cfg["test_size"], 
        random_state=d_cfg["random_state"]
    )
    
    # Recombine features and targets to save out clean training and testing split dataframes
    train_df = X_train.copy()
    train_df[d_cfg["target_col"]] = y_train
    
    test_df = X_test.copy()
    test_df[d_cfg["target_col"]] = y_test
    
    # Save files to data directory so train.py finds them instantly
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_processed.csv", index=False)
    test_df.to_csv("data/test_processed.csv", index=False)
    print("Data preprocessed successfully! Saved train_processed.csv and test_processed.csv")