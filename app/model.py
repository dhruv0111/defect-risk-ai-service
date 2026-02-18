# app/model.py

import os
import joblib
import numpy as np
import pandas as pd


# Determine base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts")


class DefectRiskModel:
    def __init__(self):
        """
        Load trained model and scaler from artifacts directory.
        """
        self.model = joblib.load(os.path.join(ARTIFACT_PATH, "model.pkl"))
        self.scaler = joblib.load(os.path.join(ARTIFACT_PATH, "scaler.pkl"))

    def predict(self, input_data: list):
        """
        Predict defect probability for a single input sample.
        """

        # 🔹 Validate feature length
        expected_features = len(self.model.feature_names_in_)

        if len(input_data) != expected_features:
            raise ValueError(
                f"Invalid number of features. Expected {expected_features}, got {len(input_data)}"
            )

        # 🔹 Convert to DataFrame with correct feature names
        input_df = pd.DataFrame(
            [input_data],
            columns=self.model.feature_names_in_
        )

        # 🔹 Scale input
        scaled_input = self.scaler.transform(input_df)

        # 🔹 Predict probability of defect (class = 1)
        probability = self.model.predict_proba(scaled_input)[0][1]

        # 🔹 Assign risk level
        if probability >= 0.7:
            risk_level = "HIGH"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "defect_probability": round(float(probability), 4),
            "risk_level": risk_level
        }
