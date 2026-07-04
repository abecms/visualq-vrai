"""Tests for evaluation harness."""

from datetime import datetime, timezone

import pandas as pd

from vrt_triage.eval.harness import evaluate_group_holdout, evaluate_heuristic
from vrt_triage.schema.feature_spec import FEATURE_COLUMNS, LABEL_COLUMN


def _synthetic_df(n: int = 60) -> pd.DataFrame:
    rows = []
    classes = ["intentional_redesign", "accidental_regression", "platform_constraint"]
    for i in range(n):
        row = {c: float(i % 5) for c in FEATURE_COLUMNS}
        row[LABEL_COLUMN] = classes[i % 3]
        row["projectId"] = "p1" if i < n * 2 // 3 else "p2"
        row["createdAt"] = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
        if classes[i % 3] == "platform_constraint":
            row["only_this_browser"] = 1.0
            row["isolated_pixel_ratio"] = 0.25
        if classes[i % 3] == "accidental_regression":
            row["interactive_elem_changed"] = 1.0
            row["mismatch_zscore"] = 3.0
        if classes[i % 3] == "intentional_redesign":
            row["run_fail_fraction"] = 0.8
        rows.append(row)
    return pd.DataFrame(rows)


def test_heuristic_eval_runs():
    metrics = evaluate_heuristic(_synthetic_df())
    assert metrics.n_train > 0
    assert metrics.n_test > 0
    assert metrics.macro_f1 >= 0.0


def test_group_holdout_runs():
    metrics = evaluate_group_holdout(_synthetic_df())
    assert metrics.n_test > 0
