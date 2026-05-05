from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_service import ModelService

app = FastAPI(title="Dermatology Model API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ModelService()


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictionItem(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    top: list[PredictionItem]
    alert: bool
    model: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return service.metadata()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict[str, object]:
    try:
        return service.predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
