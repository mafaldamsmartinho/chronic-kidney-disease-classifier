import joblib
import pandas as pd
from api.schemas import ClassificationResponse, PatientClassifyRequest
from fastapi import APIRouter
from src.predict import predict

router = APIRouter()

model = joblib.load("models/ckd_model.joblib")

@router.post("/patient")
def classify_patient_router(
    request: PatientClassifyRequest
     ) -> ClassificationResponse:
    """Classify patient ckd."""
    patient_data = pd.DataFrame([request.model_dump()])
    result = predict(model=model, data=patient_data)
    print("Classification Successfull")
    if result["prediction"].iloc[0] == 0:
        outcome = "notckd"
    else:
        outcome = "ckd"
    return ClassificationResponse(
        prediction=outcome)
