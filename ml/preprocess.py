# ml/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate input features (X) and target variable (y).
    Works across multiple NASA datasets.
    """

    # 🔎 Detect target column automatically
    if "defects" in df.columns:
        target_column = "defects"
    elif "bug" in df.columns:
        target_column = "bug"
    else:
        raise ValueError("No valid target column found in dataset.")

    # 🔎 Drop ID column only if it exists
    columns_to_drop = [target_column]
    if "id" in df.columns:
        columns_to_drop.append("id")

    X = df.drop(columns=columns_to_drop)
    y = df[target_column]

    return X, y


def scale_features(X):
    """
    Standardize numerical features using StandardScaler.
    Returns scaled features and fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
