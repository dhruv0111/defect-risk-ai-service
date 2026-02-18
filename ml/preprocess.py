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
    """
    X = df.drop(columns=["id", "defects"])
    y = df["defects"]
    return X, y


def scale_features(X):
    """
    Standardize numerical features using StandardScaler.
    Returns scaled features and fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
