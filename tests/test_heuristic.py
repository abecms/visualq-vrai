"""Tests for heuristic baseline."""


from visualq_vrai.heuristic.triage import triage_heuristic
from visualq_vrai.schema.bundle import TriageClass


def _row(**overrides):
    base = {k: float("nan") for k in [
        "mismatch_zscore", "isolated_pixel_ratio", "only_this_browser",
        "run_fail_fraction", "interactive_elem_changed", "spatial_recurrence_iou",
        "commit_changed_since_baseline", "figma_alignment_delta",
    ]}
    base.update(overrides)
    return base


def test_platform_mono_browser():
    result = triage_heuristic(_row(only_this_browser=1.0, isolated_pixel_ratio=0.2))
    assert result.class_name == TriageClass.PLATFORM.value


def test_regression_interactive_high_zscore():
    result = triage_heuristic(_row(interactive_elem_changed=1.0, mismatch_zscore=3.0))
    assert result.class_name == TriageClass.REGRESSION.value
    assert result.confidence >= 0.5


def test_intentional_broad_failure():
    result = triage_heuristic(_row(run_fail_fraction=0.8, figma_alignment_delta=0.1))
    assert result.class_name == TriageClass.INTENTIONAL.value


def test_source_is_heuristic():
    result = triage_heuristic(_row())
    assert result.source == "heuristic"
