import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score

from preprocess import load_data, split_features_target, scale_features

DATASETS = ["cm1.csv", "kc1.csv", "pc1.csv", "jm1.csv"]
DATA_PATH = "data"
RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)

def evaluate_dataset(file_name):
    print(f"\n🔹 Evaluating {file_name}")

    df = load_data(os.path.join(DATA_PATH, file_name))
    X, y = split_features_target(df)
    X_scaled, scaler = scale_features(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    auc_scores = []
    recall_scores = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)

        auc_scores.append(roc_auc_score(y_test, y_prob))
        recall_scores.append(recall_score(y_test, y_pred))

    return {
        "Dataset": file_name,
        "Mean_AUC": round(np.mean(auc_scores), 4),
        "Std_AUC": round(np.std(auc_scores), 4),
        "Mean_Recall_Defect": round(np.mean(recall_scores), 4)
    }


def run_multi_dataset():
    results = []

    for dataset in DATASETS:
        result = evaluate_dataset(dataset)
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "multi_dataset_results.csv"), index=False)

    print("\n✅ Multi-dataset results saved.")
    print(results_df)


if __name__ == "__main__":
    run_multi_dataset()