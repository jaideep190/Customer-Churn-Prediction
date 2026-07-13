"""
Data preprocessing for Telco Customer Churn dataset.
Handles missing values, encoding, and feature preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn dataset."""
    return pd.read_excel(path)


def fix_missing_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Total Charges' has blank values for new customers.
    Reconstruct using Monthly Charges * Tenure Months where missing.
    """
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    calc_charges = df['Monthly Charges'] * df['Tenure Months']
    df['Total Charges'] = np.where(
        df['Total Charges'].isna(), calc_charges, df['Total Charges']
    )
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier, leakage, and geography columns not used for modeling."""
    cols_to_drop = [
        'Country', 'State', 'Count', 'Zip Code', 'Churn Reason', 'City',
        'Churn Score', 'Churn Value', 'CLTV', 'CustomerID', 'Lat Long',
        'Latitude', 'Longitude'
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Churn Label to binary 0/1."""
    df['Churn Label'] = df['Churn Label'].replace({'Yes': 1, 'No': 0})
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all categorical (text-type) columns.
    Works across pandas versions (object dtype or new string dtype).
    Returns the encoded dataframe and a dict of fitted encoders.
    """
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def preprocess(raw_path: str) -> tuple[pd.DataFrame, dict]:
    """Full preprocessing pipeline. Returns cleaned df + encoders."""
    df = load_raw_data(raw_path)
    df = fix_missing_total_charges(df)
    df = drop_unused_columns(df)
    df = encode_target(df)
    df, encoders = encode_categoricals(df)
    return df, encoders


if __name__ == "__main__":
    df, encoders = preprocess("data/raw/Telco_customer_churn.xlsx")
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print(f"Saved cleaned data: {df.shape}")