import joblib
import pandas as pd
from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse

from api.schemas import ClassificationResponse, PatientClassifyRequest
from src.predict import predict

router = APIRouter()
app = FastAPI()

model = joblib.load("models/ckd.joblib")


@router.get("/")
def home():
    return FileResponse("templates/index.html")


@router.post("/predict")
def classify_patient_router(request: PatientClassifyRequest) -> ClassificationResponse:
    """Classify patient ckd."""
    values = request.model_dump()
    empty_count = sum(
        value is None or (isinstance(value, str) and not value.strip())
        for value in values.values()
    )
    # Application completeness rule, not a calibrated confidence threshold.
    if empty_count / len(values) >= 0.5:
        return ClassificationResponse(prediction="Undefined")

    patient_data = pd.DataFrame([values])
    # Converts the data received by FastAPI into a one-row pandas DataFrame
    result = predict(model=model, data=patient_data)
    print("Classification Successfull")
    if result["prediction"].iloc[0] == 0:
        outcome = "Not CKD"
    else:
        outcome = "CKD"

    print(outcome)
    return ClassificationResponse(prediction=outcome)


app.include_router(router)
