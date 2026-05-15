from pydantic import BaseModel


class TopPrediction(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    top5: list[TopPrediction]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
