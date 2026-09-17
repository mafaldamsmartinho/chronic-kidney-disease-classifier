import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, model
from api.schemas import PatientClassifyRequest
from src.preprocessing import CATEGORICAL_COLUMNS


@pytest.fixture
def patient():
    return {
        "age": 45,
        "bp": 80,
        "sg": 1.020,
        "al": 0,
        "su": 0,
        "bgr": 100,
        "bu": 30,
        "sc": 1.0,
        "sod": 140,
        "pot": 4.5,
        "hemo": 15,
        "pcv": 45,
        "wc": 8000,
        "rc": 5,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "htn": "no",
        "dm": "no",
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", -20),
        ("al", 99),
        ("su", 1.5),
        ("sg", 1.012),
        ("htn", "banana"),
        ("pcv", 101),
        ("sc", "inf"),
    ],
)
def test_invalid_patient_rejected(patient, field, value):
    response = TestClient(app).post("/predict", json={**patient, field: value})
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])


def test_missing_categories_match_training(patient):
    columns = CATEGORICAL_COLUMNS + ["rbc"]
    training_row = pd.DataFrame([{**patient, **dict.fromkeys(columns, np.nan)}])
    expected_encoding = model[:-1].transform(training_row)
    expected_label = "CKD" if model.predict(training_row)[0] == 1 else "Not CKD"
    for missing in [None, "", "   "]:
        payload = {**patient, **dict.fromkeys(columns, missing)}
        request = PatientClassifyRequest(**payload)
        np.testing.assert_array_equal(
            model[:-1].transform(pd.DataFrame([request.model_dump()])),
            expected_encoding,
        )
        response = TestClient(app).post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json() == {"prediction": expected_label}


@pytest.mark.parametrize("missing_count", [11, 12, 13])
def test_missingness_boundary(patient, missing_count):
    payload = {**patient, **dict.fromkeys(list(patient)[:missing_count], None)}
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    if missing_count >= 12:
        assert prediction == "Undefined"
    else:
        assert prediction in {"CKD", "Not CKD"}
