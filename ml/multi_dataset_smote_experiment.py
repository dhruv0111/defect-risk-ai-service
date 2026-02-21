import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE

from preprocess import load_data, split_features_target, scale_features

DATASETS = ["cm1.csv", "kc1.csv", "pc1.csv", "jm1.csv"]
DATA_PATH = "data"
RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)


def evaluate_strategy(file_name, strategy):
    df = load_data(os.path.join(DATA_PATH, file_name))
    X, y = split_features_target(df)
    X_scaled, scaler = scale_features(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []
    recall_scores = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if strategy == "smote":
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

            model = LogisticRegression(max_iter=1000, random_state=42)

        elif strategy == "class_weight":
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )

        else:  # no balancing
            model = LogisticRegression(max_iter=1000, random_state=42)

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)

        auc_scores.append(roc_auc_score(y_test, y_prob))
        recall_scores.append(recall_score(y_test, y_pred))

    return np.mean(auc_scores), np.mean(recall_scores)


def run_smote_study():
    results = []

    for dataset in DATASETS:
        print(f"\n🔹 Evaluating {dataset}")

        for strategy in ["none", "class_weight", "smote"]:
            auc, recall = evaluate_strategy(dataset, strategy)

            results.append({
                "Dataset": dataset,
                "Strategy": strategy,
                "Mean_AUC": round(auc, 4),
                "Mean_Recall_Defect": round(recall, 4)
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "imbalance_strategy_results.csv"), index=False)

    print("\n✅ Imbalance strategy comparison saved.")
    print(results_df)


if __name__ == "__main__":
    run_smote_study()