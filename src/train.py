"""Train and evaluate the SVM chosen in the notebook."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from src.predict import predict
from src.preprocessing import build_preprocessor, clean_data


def train(data_path="kidney_disease .csv", model_path="models/ckd.joblib"):
    data = pd.read_csv(data_path)
    target = data["classification"].str.strip().map({"notckd": 0, "ckd": 1})
    if target.isna().any():
        raise ValueError("The target must contain only ckd or notckd.")

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns="classification"),
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model = Pipeline(
        [
            ("cleaning", FunctionTransformer(clean_data)),
            ("preprocessing", build_preprocessor()),
            ("model", SVC()),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = predict(model, X_test)["prediction"]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="kidney_disease .csv")
    parser.add_argument("--model", default="models/ckd.joblib")
    args = parser.parse_args()

    for name, value in train(args.data, args.model).items():
        print(f"{name}: {value:.4f}")
