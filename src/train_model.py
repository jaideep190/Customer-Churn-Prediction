"""
Train and compare multiple churn prediction models, including a
custom focal loss variant, for the customer churn prediction project.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb


def load_processed_data(path: str):
    df = pd.read_csv(path)
    object_cols = df.select_dtypes(include='object').columns.tolist()
    if object_cols:
        raise ValueError(
            f"Found unencoded text columns: {object_cols}. "
            f"Did you run 'python src/data_preprocessing.py' first?"
        )
    x = df.drop("Churn Label", axis=1)
    y = df["Churn Label"]
    return x, y


def split_data(x, y, test_size=0.2, random_state=2):
    """Split BEFORE any resampling — avoids data leakage."""
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def get_model_configs(y_train):
    """
    Returns {model_name: (model_instance, needs_smote)}.
    needs_smote=True means the model has no native imbalance handling,
    so SMOTE is applied to its training data instead.
    """
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

    configs = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=2),
            False
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=2, n_jobs=-1
            ),
            False
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(n_neighbors=15),
            True
        ),
        "XGBoost": (
            XGBClassifier(
                learning_rate=0.01, max_depth=3, n_estimators=1000,
                scale_pos_weight=scale_weight, eval_metric="logloss", random_state=2
            ),
            False
        ),
        "LightGBM": (
            LGBMClassifier(
                learning_rate=0.01, max_depth=3, n_estimators=1000,
                class_weight="balanced", random_state=2, verbose=-1
            ),
            False
        ),
    }
    return configs


def train_all_models(x_train, y_train):
    """Train every configured sklearn-style model."""
    trained_models = {}
    configs = get_model_configs(y_train)

    over = SMOTE(sampling_strategy=1, random_state=2)
    x_train_smote, y_train_smote = over.fit_resample(x_train, y_train)

    for name, (model, needs_smote) in configs.items():
        print(f"Training {name}...")
        if needs_smote:
            model.fit(x_train_smote, y_train_smote)
        else:
            model.fit(x_train, y_train)
        trained_models[name] = model

    return trained_models


def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    x, y = load_processed_data("data/processed/cleaned_data.csv")
    x_train, x_test, y_train, y_test = split_data(x, y)

    # --- Train sklearn-style models ---
    models = train_all_models(x_train, y_train)
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        save_model(model, f"models/{safe_name}.pkl")

    # --- Save test split for evaluation ---
    x_test.to_csv("data/processed/x_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("\nAll models trained and saved to models/")