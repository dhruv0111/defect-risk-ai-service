import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    auc_score = roc_auc_score(y_test, y_prob)
    print(f"AUC Score: {round(auc_score, 4)}")

    return y_prob


def find_best_threshold(y_test, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

    print(f"Best Threshold (F1 optimized): {round(best_threshold,4)}")

    return best_threshold


def plot_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
