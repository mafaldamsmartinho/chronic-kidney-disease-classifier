from pydantic import BaseModel


class PatientClassifyRequest(BaseModel):

    age: float | None = None
    bp: float | None = None
    sg: float | None = None
    al: float | None = None
    su: float | None = None

    bgr: float | None = None
    bu: float | None = None
    sc: float | None = None
    sod: float | None = None
    pot: float | None = None
    hemo: float | None = None
    pcv: float | None = None
    wc: float | None = None
    rc: float | None = None

    rbc: str | None = None
    pc: str | None = None
    pcc: str | None = None
    ba: str | None = None
    htn: str | None = None
    dm: str | None = None
    cad: str | None = None
    appet: str | None = None
    pe: str | None = None
    ane: str | None = None


class ClassificationResponse(BaseModel):
    prediction: str