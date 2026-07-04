"""Integration test for TabICL predict path (slow)."""

import json
from pathlib import Path

import pytest

from visualq_vrai.features.extract import bundles_to_dataframe
from visualq_vrai.schema.bundle import DiffBundle
from visualq_vrai.schema.feature_spec import LABEL_COLUMN, MIN_CONTEXT_ROWS
from visualq_vrai.service.predict import predict_tabicl

pytestmark = pytest.mark.slow


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[1].joinpath("fixtures/bundles/synthetic-0000.json").exists(),
    reason="synthetic fixtures missing",
)
def test_tabicl_predict_on_synthetic_fixtures():
    bundles_dir = Path(__file__).resolve().parents[1] / "fixtures" / "bundles"
    bundles = [
        DiffBundle.model_validate(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(bundles_dir.glob("synthetic-*.json"))
    ]
    df = bundles_to_dataframe(bundles)
    labeled = df.dropna(subset=[LABEL_COLUMN])
    assert len(labeled) >= MIN_CONTEXT_ROWS

    context_df = labeled.iloc[:-1]
    query_df = labeled.iloc[-1:]
    results = predict_tabicl(context_df, query_df)
    assert len(results) == 1
    assert results[0].source == "tabicl"
    assert sum(results[0].probabilities.values()) == pytest.approx(1.0, abs=0.05)
