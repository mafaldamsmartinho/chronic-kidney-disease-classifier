from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PatientClassifyRequest(BaseModel):
    age: NonNegative | None = None
    bp: NonNegative | None = None
    sg: Literal[1.005, 1.010, 1.015, 1.020, 1.025] | None = None
    al: Literal[0, 1, 2, 3, 4, 5] | None = None
    su: Literal[0, 1, 2, 3, 4, 5] | None = None

    bgr: NonNegative | None = None
    bu: NonNegative | None = None
    sc: NonNegative | None = None
    sod: NonNegative | None = None
    pot: NonNegative | None = None
    hemo: NonNegative | None = None
    pcv: Annotated[NonNegative, Field(le=100)] | None = None
    wc: NonNegative | None = None
    rc: NonNegative | None = None

    rbc: Literal["normal", "abnormal"] | None = None
    pc: Literal["normal", "abnormal"] | None = None
    pcc: Literal["present", "notpresent"] | None = None
    ba: Literal["present", "notpresent"] | None = None
    htn: Literal["yes", "no"] | None = None
    dm: Literal["yes", "no"] | None = None
    cad: Literal["yes", "no"] | None = None
    appet: Literal["good", "poor"] | None = None
    pe: Literal["yes", "no"] | None = None
    ane: Literal["yes", "no"] | None = None

    @field_validator("*", mode="before")
    @classmethod
    def normalize_blanks(cls, value):
        if isinstance(value, str):
            return value.strip() or None
        return value


class ClassificationResponse(BaseModel):
    prediction: str
