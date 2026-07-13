"""
Train churn prediction models using two imbalance-handling strategies:
SMOTE and scale_pos_weight. Saves both models for comparison.
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def load_processed_data(path: str):
    df = pd.read_csv(path)
    x = df.drop("Churn Label", axis=1)
    y = df["Churn Label"]
    return x, y


def split_data(x, y, test_size=0.2, random_state=2):
    """Split BEFORE any resampling — avoids data leakage."""
    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_smote_model(x_train, y_train, random_state=2):
    """Approach A: SMOTE oversampling on training data only."""
    over = SMOTE(sampling_strategy=1, random_state=random_state)
    x_res, y_res = over.fit_resample(x_train, y_train)

    model = XGBClassifier(
        learning_rate=0.01, max_depth=3, n_estimators=1000,
        eval_metric='logloss'
    )
    model.fit(x_res, y_res)
    return model


def train_weighted_model(x_train, y_train):
    """Approach B: scale_pos_weight — no synthetic data."""
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        learning_rate=0.01, max_depth=3, n_estimators=1000,
        scale_pos_weight=scale_weight, eval_metric='logloss'
    )
    model.fit(x_train, y_train)
    return model


def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    x, y = load_processed_data("data/processed/cleaned_data.csv")
    x_train, x_test, y_train, y_test = split_data(x, y)

    print("Training SMOTE model...")
    smote_model = train_smote_model(x_train, y_train)
    save_model(smote_model, "models/xgb_smote.pkl")

    print("Training scale_pos_weight model...")
    weighted_model = train_weighted_model(x_train, y_train)
    save_model(weighted_model, "models/xgb_weighted.pkl")

    # Save test split for evaluation step
    x_test.to_csv("data/processed/x_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Done. Models saved to models/")