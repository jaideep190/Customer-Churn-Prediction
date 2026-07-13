
# Customer Churn Prediction & Retention ROI Simulator

An end-to-end ML project that predicts customer churn, explains individual predictions using SHAP, and simulates whether a retention offer is financially worth sending to a specific customer.

**Live Demo:** [Streamlit App Link]

![Dashboard Overview](images/streamlit_ui_1.png)

---

## Problem Statement

Customer churn directly impacts recurring revenue, and retaining an existing customer is generally cheaper than acquiring a new one. This project goes beyond a churn prediction score to answer three questions:

1. Which customers are likely to churn?
2. Why is a specific customer at risk?
3. Is it financially worth sending that customer a retention offer?

---

## Dataset

[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features covering demographics, account details, and subscribed services. Churn rate: ~26.5%.

---

## Key EDA Insights

- Month-to-month contracts churn far more than annual contracts
- ~50% of churned customers leave within the first 10 months
- Fiber optic customers churn more than DSL, likely driven by pricing
- Customers without online security or tech support churn more
- Electronic check payment method has the highest churn rate
- Senior citizens churn at nearly 2x the rate of non-seniors

---

## Modeling Approach

Class imbalance handled using `scale_pos_weight` in XGBoost, tuned against `SMOTE` as a comparison. Final model optimizes for recall over raw accuracy, since missing an actual churner is costlier to the business than a false alarm.

| Metric | Score |
|---|---|
| ROC AUC | 87.3% |
| Accuracy | 76.6% |
| Churn Recall | 0.87 |
| Churn Precision | 0.54 |

![ROC Curve](images/roc_curve.png)
![Confusion Matrix](images/confusion_matrix_weighted.png)

Threshold is adjustable in the app to shift the precision/recall trade-off based on business cost assumptions.

---

## Explainability with SHAP

Every prediction is broken down using SHAP `TreeExplainer`, showing exactly which features increased or decreased a specific customer's churn risk — not just a global feature importance chart.

![SHAP Explanation](images/shap_summary.png)

---

## Retention ROI Simulator

Connects the model output to a business decision:

```
Expected Revenue Saved = Churn Probability × Offer Success Rate × Monthly Revenue × Retained Months
Net Value = Expected Revenue Saved − Offer Cost
```

Users can adjust offer cost, success rate, and retention duration to see whether a retention offer is worth sending for a given customer.

![Retention Simulator](images/retention_simulator.png)

---

## How to Run Locally

```bash
git clone [your-repo-url]
cd churn-prediction-retention-simulator

python -m venv myenv
myenv\Scripts\activate

pip install -r requirements.txt

python src/data_preprocessing.py
python src/train_model.py
python src/evaluate.py
python src/shap_explain.py

streamlit run app/streamlit_app.py
```

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn, SHAP, Matplotlib, Seaborn, Streamlit

---
