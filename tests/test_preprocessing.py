import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing import (
    CATEGORICAL_COLUMNS,
    DISCRETE_COLUMNS,
    NUMERIC_COLUMNS,
    build_preprocessor,
    clean_data,
    mean_columns,
    median_columns,
)


def test_clean_data():
    data = pd.DataFrame(
        {
            **{column: ["1"] for column in NUMERIC_COLUMNS + DISCRETE_COLUMNS},
            **{column: ["yes"] for column in CATEGORICAL_COLUMNS},
            "id": [1],
            "classification": ["ckd"],
            "age": [" 45 "],
            "bp": ["80"],
            "rbc": [" normal "],
            "htn": [" yes "],
        }
    )

    cleaned = clean_data(data)

    # Removed columns
    assert "id" not in cleaned.columns
    assert "classification" not in cleaned.columns

    # Numeric conversion
    assert cleaned["age"].iloc[0] == 45.0
    assert cleaned["bp"].iloc[0] == 80.0
    assert cleaned["age"].dtype == float

    # Categorical whitespace removed
    assert cleaned["rbc"].iloc[0] == "normal"
    assert cleaned["htn"].iloc[0] == "yes"


def test_mean_median_columns():

    data = pd.DataFrame(
        {
            **{column: [48, 50, 52] for column in NUMERIC_COLUMNS},
            "age": [48, 50, 52],  # fairly symmetric → mean
            "bp": [2, 4, 50],  # skewed → median
        }
    )

    mean_cols = mean_columns(data)
    median_cols = median_columns(data)

    assert "age" in mean_cols
    assert "bp" in median_cols


def test_build_preprocessor():

    preprocessor = build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)

    transformer_names = [name for name, _, _ in preprocessor.transformers]

    assert transformer_names == [
        "mean",
        "median",
        "categorical",
        "discrete",
        "rbc",
    ]
