# ml/train.py

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocess import load_data, split_features_target, scale_features


DATA_PATH = "data/cm1.csv"
ARTIFACT_PATH = "artifacts"


def train_model():
    """
    Full training pipeline:
    - Load data
    - Preprocess
    - Train model
    - Save artifacts
    """

    # 1. Load dataset
    df = load_data(DATA_PATH)

    # 2. Split features and target
    X, y = split_features_target(df)

    # 3. Scale features
    X_scaled, scaler = scale_features(X)

    # 4. Train-test split (for evaluation only)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Initialize model
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    # 6. Train model
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 8. Ensure artifact directory exists
    os.makedirs(ARTIFACT_PATH, exist_ok=True)

    # 9. Save model & scaler
    joblib.dump(model, os.path.join(ARTIFACT_PATH, "model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_PATH, "scaler.pkl"))

    print("\nModel and scaler saved successfully.")


if __name__ == "__main__":
    train_model()
