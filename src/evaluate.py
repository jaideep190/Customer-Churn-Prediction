"""
Evaluate trained models: metrics, confusion matrix, ROC curve,
and threshold comparison.
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    confusion_matrix, roc_curve
)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, x_test, y_test, threshold=0.5, save_path=None):
    """Evaluate a model at a given threshold, print metrics, save confusion matrix plot."""
    y_proba = model.predict_proba(x_test)[:, 1]
    predictions = (y_proba >= threshold).astype(int)

    c_matrix = confusion_matrix(y_test, predictions)
    percentages = (c_matrix / np.sum(c_matrix, axis=1)[:, np.newaxis]).round(2) * 100
    labels = [[f"{c_matrix[i, j]} ({percentages[i, j]:.2f}%)"
               for j in range(c_matrix.shape[1])] for i in range(c_matrix.shape[0])]
    labels = np.asarray(labels)

    plt.figure()
    sns.heatmap(c_matrix, annot=labels, fmt='', cmap='Blues')
    plt.title(f"Confusion Matrix (threshold={threshold})")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, predictions)

    print(f"--- Threshold: {threshold} ---")
    print(f"ROC AUC (proba-based): {auc:.2%}")
    print(f"Model accuracy: {acc:.2%}")
    print(classification_report(y_test, predictions))

    return y_proba, auc, acc


def plot_roc_comparison(y_test, proba_dict: dict, save_path=None):
    """
    Compare ROC curves for multiple models.
    proba_dict: {"SMOTE": proba_array, "scale_pos_weight": proba_array}
    """
    plt.figure(figsize=(6, 6))
    for label, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    x_test = pd.read_csv("data/processed/x_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    smote_model = load_model("models/xgb_smote.pkl")
    weighted_model = load_model("models/xgb_weighted.pkl")

    print("=== SMOTE Model ===")
    proba_smote, _, _ = evaluate_model(
        smote_model, x_test, y_test, threshold=0.5,
        save_path="images/confusion_matrix_smote.png"
    )

    print("\n=== scale_pos_weight Model ===")
    proba_weighted, _, _ = evaluate_model(
        weighted_model, x_test, y_test, threshold=0.5,
        save_path="images/confusion_matrix_weighted.png"
    )

    print("\n=== Threshold tuning (scale_pos_weight) ===")
    for t in [0.3, 0.4, 0.5, 0.6]:
        evaluate_model(weighted_model, x_test, y_test, threshold=t)

    plot_roc_comparison(
        y_test,
        {"SMOTE": proba_smote, "scale_pos_weight": proba_weighted},
        save_path="images/roc_curve.png"
    )