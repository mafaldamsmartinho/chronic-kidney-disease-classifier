"""Cleaning and preprocessing used by both training and prediction."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
]
CATEGORICAL_COLUMNS = [
    "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
]
DISCRETE_COLUMNS = ["sg", "al", "su"]


def clean_data(data):
    """Normalize raw CSV values without removing patient records."""
    data = data.drop(columns=["id", "classification"], errors="ignore").copy()
    for column in NUMERIC_COLUMNS + DISCRETE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(float)
    for column in CATEGORICAL_COLUMNS + ["rbc"]:
        data[column] = data[column].astype(object).str.strip()
    return data


def mean_columns(data):
    skewness = data[NUMERIC_COLUMNS].skew()
    return skewness.index[skewness.abs() < 0.5].tolist()


def median_columns(data):
    return [column for column in NUMERIC_COLUMNS if column not in mean_columns(data)]


def build_preprocessor():
    """Learn imputation and encoding from training observations only."""
    def numeric_pipeline(strategy):
        return Pipeline([
            ("imputer", SimpleImputer(strategy=strategy, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])

    def categorical_pipeline(strategy="most_frequent"):
        return Pipeline([
            ("imputer", SimpleImputer(
                strategy=strategy, fill_value="unknown", keep_empty_features=True,
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    return ColumnTransformer([
        ("mean", numeric_pipeline("mean"), mean_columns),
        ("median", numeric_pipeline("median"), median_columns),
        ("categorical", categorical_pipeline(), CATEGORICAL_COLUMNS),
        ("discrete", categorical_pipeline(), DISCRETE_COLUMNS),
        ("rbc", categorical_pipeline("constant"), ["rbc"]),
    ])
