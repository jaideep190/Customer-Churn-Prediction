"""
Customer Churn Prediction & Retention ROI Simulator
Interactive Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="Churn Prediction & Retention Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    with open("models/xgb_weighted.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_test_data():
    x_test = pd.read_csv("data/processed/x_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return x_test, y_test


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


model = load_model()
x_test, y_test = load_test_data()
explainer = get_explainer(model)

# ---------------- Header ----------------
st.title("Customer Churn Prediction & Retention ROI Simulator")
st.markdown(
    "This tool predicts whether a customer is likely to churn, explains the "
    "specific factors driving that prediction for the selected customer, and "
    "estimates whether a retention offer would be financially worthwhile."
)
st.divider()

# ---------------- Sidebar ----------------
st.sidebar.header("Select a Customer")
st.sidebar.markdown(
    "Choose a customer from the test set. All numbers below update "
    "automatically based on this selection."
)
customer_idx = st.sidebar.slider("Customer index", 0, len(x_test) - 1, 0)
customer_row = x_test.iloc[[customer_idx]]
actual_label = y_test.iloc[customer_idx]

with st.sidebar.expander("How this model works"):
    st.markdown(
        "- Model: XGBoost classifier\n"
        "- Class imbalance handled using scale_pos_weight (no synthetic data)\n"
        "- Explanations generated using SHAP (SHapley Additive exPlanations), "
        "which attributes the prediction to individual feature contributions\n"
        "- Test accuracy: 76.6 percent, ROC AUC: 87.3 percent"
    )

# ---------------- Prediction Summary ----------------
churn_proba = model.predict_proba(customer_row)[0, 1]
predicted_label = "Will Churn" if churn_proba >= 0.5 else "Will Stay"
actual_text = "Churned" if actual_label == 1 else "Stayed"

st.subheader("Prediction Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{churn_proba:.1%}")
col2.metric("Model Prediction", predicted_label)
col3.metric("Actual Outcome", actual_text)

st.caption(
    "Churn probability is the model's estimated likelihood this customer "
    "stops using the service. Predictions above 50 percent are classified "
    "as likely to churn. Actual outcome is shown for reference since this "
    "customer comes from a labeled test set."
)

st.divider()

# ---------------- SHAP Explanation ----------------
st.subheader("Why This Prediction Was Made")
st.markdown(
    "The chart below shows the features that had the largest effect on this "
    "customer's churn probability. Bars extending right increased churn risk; "
    "bars extending left decreased it. Bar length reflects the size of the effect."
)

shap_values = explainer.shap_values(customer_row)
values = shap_values[0] if len(np.array(shap_values).shape) == 2 else shap_values[0][0]

contributions = pd.DataFrame({
    "Feature": customer_row.columns,
    "SHAP Value": values,
    "Customer Value": customer_row.iloc[0].values
}).sort_values("SHAP Value", key=abs, ascending=False).head(8)

fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
colors = ["#c0392b" if v > 0 else "#27ae60" for v in contributions["SHAP Value"]]
ax.barh(contributions["Feature"], contributions["SHAP Value"], color=colors, height=0.6)
ax.set_xlabel("Effect on churn probability", fontsize=9)
ax.tick_params(axis='both', labelsize=8)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()

left_col, right_col = st.columns([2, 1])
with left_col:
    st.pyplot(fig, use_container_width=False)
with right_col:
    st.markdown("**Top factor detail**")
    top_feature = contributions.iloc[0]
    direction = "increased" if top_feature["SHAP Value"] > 0 else "decreased"
    st.markdown(
        f"`{top_feature['Feature']}` had the largest effect and "
        f"{direction} this customer's churn risk."
    )
    with st.expander("View raw feature values"):
        st.dataframe(contributions, use_container_width=True, hide_index=True)

st.divider()

# ---------------- Retention ROI Simulator ----------------
st.subheader("Retention Offer Simulator")
st.markdown(
    "Adjust the assumptions below to see whether sending this customer a "
    "retention offer is expected to pay off financially."
)

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    offer_cost = st.number_input(
        "Cost of retention offer (dollars)",
        min_value=0, value=10, step=5,
        help="The discount or incentive cost to offer this customer."
    )
    monthly_revenue = st.number_input(
        "Customer's monthly revenue (dollars)",
        min_value=0.0,
        value=float(customer_row["Monthly Charges"].values[0]),
        step=5.0,
        help="Pre-filled from this customer's current monthly charges."
    )

with sim_col2:
    offer_success_rate = st.slider(
        "Estimated chance the offer prevents churn (percent)",
        0, 100, 40,
        help="Your estimate of how likely this type of offer is to retain a customer at this risk level."
    ) / 100
    retention_months = st.slider(
        "Expected retained months if offer succeeds",
        1, 24, 12,
        help="How many additional months the customer is expected to stay if retained."
    )

expected_revenue_saved = churn_proba * offer_success_rate * monthly_revenue * retention_months
net_value = expected_revenue_saved - offer_cost

if offer_cost == 0:
    st.info(
        "Offer cost is set to zero, meaning any retention action with a "
        "nonzero success rate will always appear worthwhile. Set a "
        "realistic cost (e.g., discount value, staff time) for a meaningful comparison."
    )

st.markdown("#### Result")

r1, r2, r3 = st.columns(3)
r1.metric("Expected Revenue Saved", f"${expected_revenue_saved:,.2f}")
r2.metric("Offer Cost", f"${offer_cost:,.2f}")
r3.metric("Net Value", f"${net_value:,.2f}")

if net_value > 0:
    st.success(
        f"Sending this offer is expected to net ${net_value:,.2f} in saved "
        f"revenue for this customer, based on the assumptions above."
    )
else:
    st.warning(
        f"This offer is not cost-effective for this customer under the "
        f"current assumptions (net: ${net_value:,.2f})."
    )

with st.expander("How this calculation works"):
    st.markdown(
        "Expected Revenue Saved = Churn Probability x Offer Success Rate x "
        "Monthly Revenue x Retained Months\n\n"
        "Net Value = Expected Revenue Saved - Offer Cost\n\n"
        "This is a simplified estimate intended to demonstrate how model "
        "output can be connected to a business decision. A production "
        "version would incorporate discounting, customer lifetime value, "
        "and offer-specific historical success rates."
    )

st.divider()
st.caption(
    "Model: XGBoost with scale_pos_weight for class imbalance. "
    "Explanations generated using SHAP TreeExplainer on the trained model."
)