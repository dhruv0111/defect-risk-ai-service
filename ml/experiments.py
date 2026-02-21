# ml/experiments.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, recall_score
from scipy.stats import ttest_rel

from preprocess import load_data, split_features_target, scale_features


DATA_PATH = "data/cm1.csv"
RESULTS_PATH = "results"


def ensure_results_dir():
    os.makedirs(RESULTS_PATH, exist_ok=True)


def run_experiments():
    ensure_results_dir()

    print("\n🔹 Loading dataset...")
    df = load_data(DATA_PATH)

    X, y = split_features_target(df)
    X_scaled, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    results = []
    auc_scores_all = {}

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        print(f"\n{name} Classification Report:\n")
        print(classification_report(y_test, y_pred))
        print(f"{name} AUC: {round(auc, 4)}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={round(auc,2)})")

        results.append({
            "Model": name,
            "AUC": round(auc, 4)
        })

        auc_scores_all[name] = y_prob

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model Comparison ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "roc_comparison.png"), dpi=300)
    plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "model_summary.csv"), index=False)

    print("\n🔹 Summary Results saved to model_summary.csv")

    return auc_scores_all, y_test


def run_kfold_experiment():
    print("\n🔹 Running 5-Fold Cross Validation...")

    df = load_data(DATA_PATH)
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

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc_scores.append(roc_auc_score(y_test, y_prob))
        recall_scores.append(recall_score(y_test, y_pred))

    cv_results = pd.DataFrame({
        "Fold_AUC": auc_scores,
        "Fold_Recall": recall_scores
    })

    cv_results.to_csv(os.path.join(RESULTS_PATH, "cv_results.csv"), index=False)

    print("Cross-validation results saved to cv_results.csv")

    return auc_scores


def analyze_feature_importance():
    print("\n🔹 Feature Importance Analysis")

    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    X_scaled, scaler = scale_features(X)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_scaled, y)

    feature_names = X.columns
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    importance_df["Absolute Importance"] = importance_df["Coefficient"].abs()
    importance_df = importance_df.sort_values(
        by="Absolute Importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(RESULTS_PATH, "feature_importance.csv"),
        index=False
    )

    top_features = importance_df.head(10)

    plt.figure(figsize=(8, 6))
    plt.barh(top_features["Feature"], top_features["Absolute Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Important Features")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "feature_importance.png"), dpi=300)
    plt.close()

    print("Feature importance saved.")


if __name__ == "__main__":
    run_experiments()
    run_kfold_experiment()
    analyze_feature_importance()
