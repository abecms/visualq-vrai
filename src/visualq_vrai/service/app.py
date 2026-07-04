"""FastAPI service for VRT diff triage."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from visualq_vrai.schema.feature_spec import SCHEMA_VERSION
from visualq_vrai.service.predict import (
    _decode_parquet,
    predict_heuristic,
    predict_tabicl,
)

app = FastAPI(title="visualq-vrai", version="0.1.0")


class PredictRequest(BaseModel):
    schemaVersion: int = Field(default=SCHEMA_VERSION)
    contextParquet: str
    queryParquet: str


class PredictResponse(BaseModel):
    results: list[dict[str, Any]]
    schemaVersion: int = SCHEMA_VERSION


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if body.schemaVersion != SCHEMA_VERSION:
        raise HTTPException(
            status_code=422,
            detail=f"schemaVersion mismatch: expected {SCHEMA_VERSION}, got {body.schemaVersion}",
        )
    context_df = _decode_parquet(body.contextParquet)
    query_df = _decode_parquet(body.queryParquet)
    raw = predict_tabicl(context_df, query_df, schema_version=body.schemaVersion)
    return PredictResponse(results=[_serialize(r) for r in raw])


@app.post("/heuristic", response_model=PredictResponse)
def heuristic(body: PredictRequest) -> PredictResponse:
    if body.schemaVersion != SCHEMA_VERSION:
        raise HTTPException(status_code=422, detail="schemaVersion mismatch")
    query_df = _decode_parquet(body.queryParquet)
    raw = predict_heuristic(query_df)
    return PredictResponse(results=[_serialize(r) for r in raw])


def _serialize(result) -> dict[str, Any]:
    payload = {
        "class": result.class_name,
        "probabilities": result.probabilities,
        "topFeatures": result.top_features,
        "conformal": result.conformal,
        "source": result.source,
    }
    if result.insufficient_data:
        payload["insufficientData"] = True
        payload["insufficientReason"] = result.insufficient_reason
    return payload
