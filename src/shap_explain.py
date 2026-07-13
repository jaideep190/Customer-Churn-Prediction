"""
SHAP explainability for the churn model.
Generates global feature importance and per-customer explanations.
"""

import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_shap_explainer(model):
    """XGBoost is tree-based, so TreeExplainer is fast and exact."""
    return shap.TreeExplainer(model)


def global_feature_importance(explainer, x_sample, save_path=None):
    """Summary plot: which features matter most across all customers."""
    shap_values = explainer.shap_values(x_sample)

    plt.figure()
    shap.summary_plot(shap_values, x_sample, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return shap_values


def explain_customer(explainer, x_row, feature_names):
    """
    Return top contributing features for a single customer's prediction.
    x_row: a single-row DataFrame (one customer).
    """
    shap_values = explainer.shap_values(x_row)

    # shap_values shape: (1, n_features) for binary classification with TreeExplainer
    values = shap_values[0] if len(shap_values.shape) == 2 else shap_values[0][0]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
        "feature_value": x_row.iloc[0].values
    }).sort_values("shap_value", key=abs, ascending=False)

    return contributions


if __name__ == "__main__":
    model = load_model("models/xgb_weighted.pkl")
    x_test = pd.read_csv("data/processed/x_test.csv")

    explainer = get_shap_explainer(model)

    # Global importance (use a sample for speed if x_test is large)
    sample = x_test.sample(min(500, len(x_test)), random_state=1)
    global_feature_importance(explainer, sample, save_path="images/shap_summary.png")
    print("Saved global SHAP summary plot to images/shap_summary.png")

    # Example: explain one customer
    example_customer = x_test.iloc[[0]]
    result = explain_customer(explainer, example_customer, x_test.columns.tolist())
    print("\nTop drivers for customer 0:")
    print(result.head(5))