"""FastAPI service for VRT diff triage."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from visualq_vrai.schema.feature_spec import SCHEMA_VERSION
from visualq_vrai.service.predict import (
    _decode_parquet,
    predict_heuristic,
    predict_tabicl,
)
from visualq_vrai.ui.routes import router as ui_router

app = FastAPI(title="visualq-vrai", version="0.1.0")
app.include_router(ui_router)


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


class PredictBundlesRequest(BaseModel):
    """Raw DiffBundle JSON payloads — featurization happens server-side.

    This is the primary integration surface for producers (e.g. VisualQ):
    they ship bundles, never tabular features. The parquet /predict endpoint
    remains for eval tooling and power users.
    """

    schemaVersion: int = Field(default=SCHEMA_VERSION)
    contextBundles: list[dict[str, Any]]
    queryBundles: list[dict[str, Any]]


@app.post("/predict-bundles", response_model=PredictResponse)
def predict_bundles(body: PredictBundlesRequest) -> PredictResponse:
    if body.schemaVersion != SCHEMA_VERSION:
        raise HTTPException(
            status_code=422,
            detail=f"schemaVersion mismatch: expected {SCHEMA_VERSION}, got {body.schemaVersion}",
        )
    if not body.queryBundles:
        raise HTTPException(status_code=422, detail="queryBundles must not be empty")

    from pydantic import ValidationError

    from visualq_vrai.features.extract import bundles_to_dataframe
    from visualq_vrai.schema.bundle import DiffBundle

    try:
        context = [DiffBundle.model_validate(b) for b in body.contextBundles]
        query = [DiffBundle.model_validate(b) for b in body.queryBundles]
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"invalid DiffBundle: {exc}") from exc

    import pandas as pd

    context_df = bundles_to_dataframe(context) if context else pd.DataFrame()
    query_df = bundles_to_dataframe(query)
    raw = predict_tabicl(context_df, query_df, schema_version=body.schemaVersion)
    return PredictResponse(results=[_serialize(r) for r in raw])


class SimilarityRequest(BaseModel):
    """Two base64-encoded images (PNG/JPEG) to compare perceptually."""

    imageA: str
    imageB: str


class SimilarityResponse(BaseModel):
    similarity: float


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(body: SimilarityRequest) -> SimilarityResponse:
    """Design-as-oracle: multi-scale pHash similarity in [0, 1].

    Callers (VisualQ) compare baseline↔design and test↔design renders to
    derive figma_alignment_baseline / figma_alignment_delta.
    """
    import base64

    from visualq_vrai.features.perceptual import perceptual_similarity

    try:
        image_a = base64.b64decode(body.imageA)
        image_b = base64.b64decode(body.imageB)
        score = perceptual_similarity(image_a, image_b)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"similarity failed: {exc}") from exc
    return SimilarityResponse(similarity=score)


@app.post("/heuristic", response_model=PredictResponse)
def heuristic(body: PredictRequest) -> PredictResponse:
    if body.schemaVersion != SCHEMA_VERSION:
        raise HTTPException(status_code=422, detail="schemaVersion mismatch")
    query_df = _decode_parquet(body.queryParquet)
    raw = predict_heuristic(query_df)
    return PredictResponse(results=[_serialize(r) for r in raw])


def create_app() -> FastAPI:
    """Return the ASGI app, honoring VRAI_URL_PREFIX for reverse proxies.

    When VRAI_URL_PREFIX is set (e.g. "/ml" behind a path-routing load
    balancer that does not strip the prefix), the service is mounted under
    that prefix so /ml/predict, /ml/health, etc. resolve.
    """
    prefix = os.environ.get("VRAI_URL_PREFIX", "").rstrip("/")
    if not prefix:
        return app
    if not prefix.startswith("/"):
        raise ValueError(f"VRAI_URL_PREFIX must start with '/': {prefix!r}")
    parent = FastAPI(title="visualq-vrai", version=app.version)
    parent.mount(prefix, app)
    return parent


def _nan_to_none(value: Any) -> Any:
    """JSON has no NaN — unknown numeric values are null on the wire."""
    import math

    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def _serialize(result) -> dict[str, Any]:
    payload = {
        "class": result.class_name,
        "probabilities": _nan_to_none(result.probabilities),
        "topFeatures": _nan_to_none(result.top_features),
        "conformal": _nan_to_none(result.conformal),
        "source": result.source,
    }
    if result.insufficient_data:
        payload["insufficientData"] = True
        payload["insufficientReason"] = result.insufficient_reason
    return payload
