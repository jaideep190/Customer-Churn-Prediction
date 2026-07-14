"""
Evaluate trained churn models across multiple thresholds.
Generates a labeled confusion matrix image for every method/threshold
combination, plus a combined ROC curve comparison.
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


def evaluate_model(model, x_test, y_test, threshold, method_name, save_path=None):
    """
    Evaluate a model at a given threshold.
    Saves a confusion matrix image titled with method name and threshold.
    Returns probabilities and a metrics dict for summary reporting.
    """
    y_proba = model.predict_proba(x_test)[:, 1]
    predictions = (y_proba >= threshold).astype(int)

    c_matrix = confusion_matrix(y_test, predictions)
    percentages = (c_matrix / np.sum(c_matrix, axis=1)[:, np.newaxis]).round(2) * 100
    labels = [[f"{c_matrix[i, j]} ({percentages[i, j]:.2f}%)"
               for j in range(c_matrix.shape[1])] for i in range(c_matrix.shape[0])]
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sns.heatmap(c_matrix, annot=labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{method_name} — Threshold {threshold}", fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    metrics = {
        "method": method_name,
        "threshold": threshold,
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "churn_precision": round(report["1"]["precision"], 4),
        "churn_recall": round(report["1"]["recall"], 4),
        "churn_f1": round(report["1"]["f1-score"], 4),
    }

    print(f"--- {method_name} | Threshold: {threshold} ---")
    print(f"ROC AUC: {auc:.2%} | Accuracy: {acc:.2%} | "
          f"Churn Precision: {metrics['churn_precision']:.2f} | "
          f"Churn Recall: {metrics['churn_recall']:.2f}")

    return y_proba, metrics


def plot_roc_comparison(y_test, proba_dict: dict, save_path=None):
    """Compare ROC curves for all model variants on a single plot."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for label, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — SMOTE vs scale_pos_weight")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    x_test = pd.read_csv("data/processed/x_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    smote_model = load_model("models/xgb_smote.pkl")
    weighted_model = load_model("models/xgb_weighted.pkl")

    all_metrics = []
    proba_for_roc = {}

    # --- SMOTE model at its standard threshold ---
    proba_smote, m = evaluate_model(
        smote_model, x_test, y_test, threshold=0.5, method_name="SMOTE",
        save_path="images/confusion_matrix_smote_t0.5.png"
    )
    all_metrics.append(m)
    proba_for_roc["SMOTE"] = proba_smote

    # --- scale_pos_weight model across multiple thresholds ---
    thresholds = [0.3, 0.4, 0.5, 0.6]
    proba_weighted = None
    for t in thresholds:
        proba_weighted, m = evaluate_model(
            weighted_model, x_test, y_test, threshold=t,
            method_name="scale_pos_weight",
            save_path=f"images/confusion_matrix_weighted_t{t}.png"
        )
        all_metrics.append(m)

    proba_for_roc["scale_pos_weight"] = proba_weighted

    # --- ROC comparison ---
    plot_roc_comparison(y_test, proba_for_roc, save_path="images/roc_curve.png")

    # --- Save all metrics as a summary table ---
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("data/processed/model_comparison.csv", index=False)

    print("\n=== Summary ===")
    print(metrics_df.to_string(index=False))
    print("\nAll images saved to images/. Metrics saved to data/processed/model_comparison.csv")