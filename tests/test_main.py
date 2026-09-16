from unittest.mock import Mock

import pandas as pd

from api import main
from api.schemas import PatientClassifyRequest


def test_too_many_missing_values(monkeypatch):
    """Should return Undefined when >= 50% of fields are missing."""

    request = PatientClassifyRequest.model_construct(
        age=50,
        bp=80,
        sg=None,
        al=None,
        su=None,
        rbc=None,
        pc=None,
        pcc=None,
        ba=None,
        bgr=None,
        bu=None,
        sc=None,
        sod=None,
        pot=None,
        hemo=None,
        pcv=None,
        wc=None,
        rc=None,
        htn=None,
        dm=None,
        cad=None,
        appet=None,
        pe=None,
        ane=None,
    )

    mock_predict = Mock()
    monkeypatch.setattr(main, "predict", mock_predict)

    response = main.classify_patient_router(request)

    assert response.prediction == "Undefined"
    mock_predict.assert_not_called()


def test_predict_not_ckd(monkeypatch):
    """Should return Not CKD when model predicts 0."""

    request = PatientClassifyRequest(
        age=45,
        bp=80,
        sg=1.020,
        al=0,
        su=0,
        rbc="normal",
        pc="normal",
        pcc="notpresent",
        ba="notpresent",
        bgr=100,
        bu=30,
        sc=1.0,
        sod=140,
        pot=4.5,
        hemo=15,
        pcv=45,
        wc=8000,
        rc=5,
        htn="no",
        dm="no",
        cad="no",
        appet="good",
        pe="no",
        ane="no",
    )

    monkeypatch.setattr(
        main,
        "predict",
        lambda model, data: pd.DataFrame({"prediction": [0]}),
    )

    response = main.classify_patient_router(request)

    assert response.prediction == "Not CKD"


def test_predict_ckd(monkeypatch):
    """Should return CKD when model predicts 1."""

    request = PatientClassifyRequest(
        age=68,
        bp=100,
        sg=1.010,
        al=3,
        su=2,
        rbc="abnormal",
        pc="abnormal",
        pcc="present",
        ba="present",
        bgr=250,
        bu=90,
        sc=4.5,
        sod=130,
        pot=5.5,
        hemo=9,
        pcv=30,
        wc=12000,
        rc=3,
        htn="yes",
        dm="yes",
        cad="yes",
        appet="poor",
        pe="yes",
        ane="yes",
    )

    monkeypatch.setattr(
        main,
        "predict",
        lambda model, data: pd.DataFrame({"prediction": [1]}),
    )

    response = main.classify_patient_router(request)

    assert response.prediction == "CKD"
