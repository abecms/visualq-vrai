"""Tests for conformal selector and predict guardrails."""

import numpy as np
import pandas as pd
import pytest

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


def test_heuristic_predict_endpoint_shape():
    row = {c: 0.0 for c in FEATURE_COLUMNS}
    row["only_this_browser"] = 1.0
    row["isolated_pixel_ratio"] = 0.3
    df = pd.DataFrame([row])
    results = predict_heuristic(df)
    assert results[0].source == "heuristic"
    assert "probabilities" in results[0].__dict__ or results[0].probabilities
