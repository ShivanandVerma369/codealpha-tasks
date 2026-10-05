"""
Data Preprocessing Module for Sales Prediction Project.

This module handles:
1. Loading the Advertising dataset.
2. Inspecting dataset structure, summary statistics, and data types.
3. Data cleaning: removing index artifacts ('Unnamed: 0'), verifying missing values, duplicates, and ranges.
4. Splitting features and target into Training and Testing sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str = "data/Advertising.csv") -> pd.DataFrame:
    """
    Loads the Advertising dataset from the specified CSV path.
    """
    if not os.path.isabs(filepath):
        # Allow relative paths whether called from project root or src folder
        if not os.path.exists(filepath):
            alt_path = os.path.join(os.path.dirname(__file__), "..", filepath)
            if os.path.exists(alt_path):
                filepath = os.path.abspath(alt_path)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded successfully from {filepath}")
    print(f"[INFO] Initial Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Prints key inspection details for the dataset in beginner-friendly format.
    """
    print("\n" + "=" * 50)
    print("--- FIRST 5 ROWS ---")
    print(df.head())

    print("\n--- LAST 5 ROWS ---")
    print(df.tail())

    print("\n--- DATASET SHAPE ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- COLUMN NAMES ---")
    print(list(df.columns))

    print("\n--- DATA TYPES & NON-NULL COUNTS ---")
    print(df.info())

    print("\n--- STATISTICAL SUMMARY ---")
    print(df.describe())
    print("=" * 50 + "\n")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning:
    - Removes index column 'Unnamed: 0' if present.
    - Validates missing values and duplicate rows.
    - Ensures all values are non-negative numeric floats.
    """
    df_clean = df.copy()

    # 1. Remove unnecessary index column
    if "Unnamed: 0" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Unnamed: 0"])
        print("[INFO] Removed unnecessary column: 'Unnamed: 0'")

    # 2. Check for missing values
    missing_counts = df_clean.isnull().sum()
    print("\n[CHECK] Missing values per column:")
    print(missing_counts)
    if missing_counts.sum() > 0:
        print("[WARNING] Missing values found! Imputing with column median.")
        df_clean = df_clean.fillna(df_clean.median())
    else:
        print("[INFO] No missing values detected.")

    # 3. Check for duplicates
    dup_count = df_clean.duplicated().sum()
    print(f"\n[CHECK] Duplicate rows count: {dup_count}")
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"[INFO] Dropped {dup_count} duplicate row(s).")
    else:
        print("[INFO] No duplicate rows detected.")

    # 4. Validate negative values
    for col in ["TV", "Radio", "Newspaper", "Sales"]:
        if col in df_clean.columns:
            neg_count = (df_clean[col] < 0).sum()
            if neg_count > 0:
                print(f"[WARNING] Found {neg_count} negative value(s) in {col}. Clipping to 0.")
                df_clean[col] = df_clean[col].clip(lower=0)

    print(f"\n[INFO] Cleaned Dataset Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    return df_clean


def prepare_features_and_target(df: pd.DataFrame):
    """
    Separates the input features (TV, Radio, Newspaper) and target variable (Sales).
    """
    feature_cols = ["TV", "Radio", "Newspaper"]
    target_col = "Sales"

    X = df[feature_cols]
    y = df[target_col]

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target into training (80%) and testing (20%) sets.
    random_state=42 ensures reproducibility of train/test partition.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Split complete: Training set ({X_train.shape[0]} samples), Test set ({X_test.shape[0]} samples)")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_df = load_data()
    inspect_data(raw_df)
    clean_df = clean_data(raw_df)
    X, y = prepare_features_and_target(clean_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
