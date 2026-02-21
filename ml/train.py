# ml/train.py

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocess import load_data, split_features_target, scale_features
from evaluate import evaluate_model, plot_roc, find_best_threshold


# Dynamically resolve project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cm1.csv")
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts")


def train_model():
    """
    Full research-grade training pipeline:
    - Load data
    - Preprocess
    - Train model
    - Evaluate with AUC & ROC
    - Optimize threshold
    - Save artifacts
    """

    print("\n🔹 Loading dataset...")
    df = load_data(DATA_PATH)

    print("🔹 Splitting features and target...")
    X, y = split_features_target(df)

    print("🔹 Scaling features...")
    X_scaled, scaler = scale_features(X)

    print("🔹 Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("🔹 Training Logistic Regression model...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\n🔹 Evaluating model performance...")
    y_prob = evaluate_model(model, X_test, y_test)

    print("\n🔹 Optimizing decision threshold...")
    best_threshold = find_best_threshold(y_test, y_prob)

    print(f"\n✅ Best Threshold Found: {round(best_threshold, 4)}")

    print("\n🔹 Plotting ROC Curve...")
    plot_roc(y_test, y_prob)

    # Save artifacts
    os.makedirs(ARTIFACT_PATH, exist_ok=True)

    joblib.dump(model, os.path.join(ARTIFACT_PATH, "model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_PATH, "scaler.pkl"))

    print("\n✅ Model and scaler saved successfully.")


if __name__ == "__main__":
    train_model()
