"""Predict CKD using the fitted SVM's default decision rule."""

import argparse

import joblib
import pandas as pd


def predict(model, data):
    """Return class labels (1 = CKD, 0 = not CKD)."""
    return pd.DataFrame({
        "prediction": model.predict(data).astype(int),
    }, index=data.index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="CSV containing the original predictor columns")
    parser.add_argument("--model", default="models/ckd.joblib")
    args = parser.parse_args()

    model = joblib.load(args.model)
    data = pd.read_csv("sample_patients.csv")
    predictions = predict(model, data)
    print(predictions.to_csv(index=False, lineterminator="\n"), end="")
