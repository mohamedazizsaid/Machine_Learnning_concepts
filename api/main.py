from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_service import ModelService

app = FastAPI(title="Dermatology Model API", version="2.0.0")

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


class FeatureMeta(BaseModel):
    name: str
    min: float
    max: float
    mean: float


class PredictionItem(BaseModel):
    label: str
    probability: float


class ClassificationMetadata(BaseModel):
    features: list[FeatureMeta]
    classes: list[str]
    model: str
    f1_macro: float


class ClassificationResponse(BaseModel):
    top: list[PredictionItem]
    alert: bool
    model: str


class SegmentationModelInfo(BaseModel):
    name: str
    k: int
    silhouette: float
    davies_bouldin: float


class SegmentationBenchmarkEntry(BaseModel):
    name: str
    clusters: int
    silhouette: float | None
    davies_bouldin: float | None
    noise: int


class SegmentationMetadata(BaseModel):
    features: list[FeatureMeta]
    model: SegmentationModelInfo
    cluster_sizes: dict[int, int]
    benchmark: list[SegmentationBenchmarkEntry]


class SegmentationResponse(BaseModel):
    cluster: int
    model: str
    k: int
    distance: float
    cluster_size: int


class FeatureImportance(BaseModel):
    name: str
    importance: float


class RecommendationMetadata(BaseModel):
    features: list[FeatureMeta]
    classes: list[str]
    model: str
    top_features: list[FeatureImportance]


class RecommendationResponse(BaseModel):
    top: list[PredictionItem]
    alert: bool
    model: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/classification/metadata", response_model=ClassificationMetadata)
def classification_metadata() -> dict[str, object]:
    return service.classification_metadata()


@app.post("/classification/predict", response_model=ClassificationResponse)
def classification_predict(request: PredictRequest) -> dict[str, object]:
    try:
        return service.classification_predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/segmentation/metadata", response_model=SegmentationMetadata)
def segmentation_metadata() -> dict[str, object]:
    return service.segmentation_metadata()


@app.post("/segmentation/predict", response_model=SegmentationResponse)
def segmentation_predict(request: PredictRequest) -> dict[str, object]:
    try:
        return service.segmentation_predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/recommendation/metadata", response_model=RecommendationMetadata)
def recommendation_metadata() -> dict[str, object]:
    return service.recommendation_metadata()


@app.post("/recommendation/predict", response_model=RecommendationResponse)
def recommendation_predict(request: PredictRequest) -> dict[str, object]:
    try:
        return service.recommendation_predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/metadata", response_model=ClassificationMetadata)
def metadata() -> dict[str, object]:
    return service.classification_metadata()


@app.post("/predict", response_model=ClassificationResponse)
def predict(request: PredictRequest) -> dict[str, object]:
    try:
        return service.classification_predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
