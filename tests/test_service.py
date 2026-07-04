"""Tests for conformal selector and predict guardrails."""

import numpy as np
import pandas as pd

from visualq_vrai.schema.feature_spec import FEATURE_COLUMNS, LABEL_COLUMN, MIN_CONTEXT_ROWS
from visualq_vrai.service.conformal import ConformalSelector
from visualq_vrai.service.predict import _check_context, predict_heuristic


def test_min_context_guard():
    df = pd.DataFrame({
        LABEL_COLUMN: ["intentional_redesign"] * 10,
        **{c: [0.0] * 10 for c in FEATURE_COLUMNS},
    })
    reason = _check_context(df)
    assert reason is not None
    assert str(MIN_CONTEXT_ROWS) in reason


def test_conformal_defers_high_regression_risk():
    selector = ConformalSelector(nominal_risk=0.02)
    probas = np.array([
        [0.1, 0.85, 0.05],
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
    ])
    y = np.array([1, 0, 0])
    selector.fit(probas, y)
    decisions = selector.decide(np.array([[0.05, 0.9, 0.05]]))
    assert decisions[0].auto_triage is False


def test_create_app_mounts_under_url_prefix(monkeypatch):
    from fastapi.testclient import TestClient

    from visualq_vrai.service.app import create_app

    monkeypatch.setenv("VRAI_URL_PREFIX", "/ml")
    client = TestClient(create_app())
    assert client.get("/ml/health").json() == {"status": "ok"}
    assert client.get("/health").status_code == 404

    monkeypatch.delenv("VRAI_URL_PREFIX")
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}


def _minimal_bundle(sample_id: str = "run__scenario__desktop__chromium") -> dict:
    return {
        "bundleVersion": 1,
        "sampleId": sample_id,
        "diff": {
            "scenario": "Home",
            "viewport": "desktop",
            "browser": "chromium",
            "status": "fail",
            "misMatchPercentage": 4.2,
            "diffPixels": 1000,
            "totalPixels": 100000,
            "dimensions": {"width": 1440, "height": 2000},
        },
        "runContext": {"runId": "run", "createdAt": "2026-07-04T10:00:00Z"},
    }


def test_predict_bundles_endpoint_featurizes_server_side():
    """Producers send raw bundles, never features — insufficient context must
    yield the heuristic verdict with JSON-safe nulls (no NaN on the wire)."""
    from fastapi.testclient import TestClient

    from visualq_vrai.service.app import app

    client = TestClient(app)
    res = client.post("/predict-bundles", json={
        "schemaVersion": 1,
        "contextBundles": [],
        "queryBundles": [_minimal_bundle()],
    })
    assert res.status_code == 200
    result = res.json()["results"][0]
    assert result["insufficientData"] is True
    assert result["source"] == "heuristic"
    assert result["class"] in {
        "intentional_redesign", "accidental_regression", "platform_constraint",
    }
    assert all(v is None for v in result["probabilities"].values())


def test_predict_bundles_endpoint_rejects_bad_input():
    from fastapi.testclient import TestClient

    from visualq_vrai.service.app import app

    client = TestClient(app)
    empty_query = client.post("/predict-bundles", json={
        "schemaVersion": 1, "contextBundles": [], "queryBundles": [],
    })
    assert empty_query.status_code == 422

    bad_bundle = client.post("/predict-bundles", json={
        "schemaVersion": 1, "contextBundles": [], "queryBundles": [{"nope": True}],
    })
    assert bad_bundle.status_code == 422

    bad_version = client.post("/predict-bundles", json={
        "schemaVersion": 99, "contextBundles": [], "queryBundles": [_minimal_bundle()],
    })
    assert bad_version.status_code == 422


def test_heuristic_predict_endpoint_shape():
    row = {c: 0.0 for c in FEATURE_COLUMNS}
    row["only_this_browser"] = 1.0
    row["isolated_pixel_ratio"] = 0.3
    df = pd.DataFrame([row])
    results = predict_heuristic(df)
    assert results[0].source == "heuristic"
    assert "probabilities" in results[0].__dict__ or results[0].probabilities
