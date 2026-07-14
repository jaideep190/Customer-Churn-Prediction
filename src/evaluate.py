"""
Evaluate and compare all trained churn models, including the focal
loss variant. Generates a labeled confusion matrix per model/threshold
and a combined ROC curve across all models.
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    confusion_matrix, roc_curve
)

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "K-Nearest Neighbors": "models/k-nearest_neighbors.pkl",
    "XGBoost": "models/xgboost.pkl",
    "LightGBM": "models/lightgbm.pkl",
}

FOCAL_MODEL_FILE = "models/xgboost_focal_loss.pkl"


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_confusion_matrix(y_test, predictions, title, save_path):
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
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def compute_metrics(y_test, y_proba, threshold, method_name):
    predictions = (y_proba >= threshold).astype(int)
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
          f"Precision: {metrics['churn_precision']:.2f} | "
          f"Recall: {metrics['churn_recall']:.2f} | F1: {metrics['churn_f1']:.2f}")

    return predictions, metrics


def evaluate_sklearn_model(model, x_test, y_test, threshold, method_name, save_path=None):
    """For standard sklearn-API models (predict_proba available)."""
    y_proba = model.predict_proba(x_test)[:, 1]
    predictions, metrics = compute_metrics(y_test, y_proba, threshold, method_name)
    if save_path:
        save_confusion_matrix(y_test, predictions, f"{method_name} — Threshold {threshold}", save_path)
    return y_proba, metrics



def plot_roc_comparison(y_test, proba_dict: dict, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    for label, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — All Models")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    x_test = pd.read_csv("data/processed/x_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    all_metrics = []
    proba_for_roc = {}

    # --- Evaluate all sklearn-style models at threshold 0.5 ---
    for name, path in MODEL_FILES.items():
        model = load_model(path)
        safe_name = name.lower().replace(" ", "_")
        proba, m = evaluate_sklearn_model(
            model, x_test, y_test, threshold=0.5, method_name=name,
            save_path=f"images/confusion_matrix_{safe_name}_t0.5.png"
        )
        all_metrics.append(m)
        proba_for_roc[name] = proba


    # --- Threshold sweep on XGBoost (scale_pos_weight variant) ---
    xgb_model = load_model(MODEL_FILES["XGBoost"])
    for t in [0.3, 0.4, 0.6]:
        _, m = evaluate_sklearn_model(
            xgb_model, x_test, y_test, threshold=t, method_name="XGBoost",
            save_path=f"images/confusion_matrix_xgboost_t{t}.png"
        )
        all_metrics.append(m)

    # --- Combined ROC curve across all models ---
    plot_roc_comparison(y_test, proba_for_roc, save_path="images/roc_curve_all_models.png")

    # --- Save summary table ---
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("data/processed/model_comparison.csv", index=False)

    print("\n=== Full Summary ===")
    print(metrics_df.to_string(index=False))
    print("\nAll images saved to images/. Metrics saved to data/processed/model_comparison.csv")