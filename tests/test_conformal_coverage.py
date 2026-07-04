"""Conformal hardening — held-out calibration, empirical coverage, fail-loud deferral.

Intent: auto-triage decisions carry a finite-sample guarantee ("at most ~2%
of auto-triaged cases are missed regressions"). That guarantee only holds if
the calibration set is disjoint from the fit context and large enough — when
it is not, the selector must defer everything, never guess a threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualq_vrai.schema.bundle import TriageClass
from visualq_vrai.service.conformal import ConformalSelector
from visualq_vrai.service.predict import (
    CALIBRATION_FRACTION,
    MIN_CALIBRATION_ROWS,
)

REG_IDX = TriageClass.index(TriageClass.REGRESSION.value)


def _synthetic_probas(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Exchangeable synthetic (probabilities, labels) with a calibrated-ish model.

    Mimics the real VRT regime: a majority of diffs are near-certainly benign
    (platform noise, approved zones) plus a risky tail. True regression
    probability drives both the label draw and the reported probability
    (plus noise) — the regime conformal calibration assumes.
    """
    is_safe = rng.random(n) < 0.6
    true_p = np.where(
        is_safe,
        rng.beta(1.0, 200.0, size=n),  # near-zero regression risk
        rng.beta(2.0, 3.0, size=n),  # risky tail, mean 0.4
    )
    labels = (rng.random(n) < true_p).astype(int) * REG_IDX
    reported = np.clip(true_p + rng.normal(0, 0.02, size=n), 0.001, 0.999)
    other = (1.0 - reported) / 2.0
    probas = np.zeros((n, 3))
    probas[:, REG_IDX] = reported
    for i in range(3):
        if i != REG_IDX:
            probas[:, i] = other
    y = np.where(labels == REG_IDX, REG_IDX, (REG_IDX + 1) % 3)
    return probas, y


def test_empirical_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(42)
    calib_probas, calib_y = _synthetic_probas(rng, 800)
    test_probas, test_y = _synthetic_probas(rng, 4000)

    selector = ConformalSelector(nominal_risk=0.02)
    selector.fit(calib_probas, calib_y)
    decisions = selector.decide(test_probas)

    auto = [i for i, d in enumerate(decisions) if d.auto_triage]
    assert len(auto) > 0, "selector auto-triaged nothing on easy synthetic data"
    missed = sum(1 for i in auto if test_y[i] == REG_IDX)
    miss_rate = missed / len(auto)
    # Finite-sample slack: nominal 2% + sampling tolerance.
    assert miss_rate <= 0.04, f"missed-regression rate {miss_rate:.3f} exceeds bound"


def test_selector_defers_when_no_safe_threshold_exists():
    # All calibration cases are regressions — no eligible cutoff at 2% risk.
    probas = np.zeros((20, 3))
    probas[:, REG_IDX] = np.linspace(0.3, 0.9, 20)
    y = np.full(20, REG_IDX)
    selector = ConformalSelector(nominal_risk=0.02)
    selector.fit(probas, y)
    decisions = selector.decide(probas)
    assert all(not d.auto_triage for d in decisions)
    assert all(d.defer_reason == "conformal_defer" for d in decisions)


def test_calibration_split_constants_are_sane():
    # Guards the contract encoded in predict_tabicl: a real held-out slice.
    assert MIN_CALIBRATION_ROWS >= 30
    assert 0.05 <= CALIBRATION_FRACTION <= 0.5


@pytest.mark.slow
def test_predict_tabicl_calibration_disjoint_and_defer_reason():
    """End-to-end: context large enough to fit but too small to also calibrate
    must serve predictions with autoTriage disabled and an explicit reason."""
    import pandas as pd

    from visualq_vrai.schema.feature_spec import (
        FEATURE_COLUMNS,
        LABEL_COLUMN,
        MIN_CONTEXT_ROWS,
    )
    from visualq_vrai.service.predict import predict_tabicl

    rng = np.random.default_rng(7)
    n = MIN_CONTEXT_ROWS + 10  # not enough to carve out MIN_CALIBRATION_ROWS
    classes = [c.value for c in TriageClass.ordered()]
    rows = []
    for i in range(n):
        cls = classes[i % 3]
        row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
        row["mismatch_zscore"] = {"accidental_regression": 3.0,
                                  "intentional_redesign": 0.5,
                                  "platform_constraint": -0.5}[cls] + rng.normal(0, 0.2)
        row[LABEL_COLUMN] = cls
        row["createdAt"] = f"2026-01-{(i % 28) + 1:02d}"
        rows.append(row)
    context = pd.DataFrame(rows)
    query = pd.DataFrame([{c: 0.0 for c in FEATURE_COLUMNS}])

    results = predict_tabicl(context, query)
    assert results[0].source == "tabicl"
    assert results[0].conformal["autoTriage"] is False
    assert results[0].conformal["deferReason"] == "calibration_too_small"
