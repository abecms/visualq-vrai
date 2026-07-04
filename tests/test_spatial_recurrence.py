"""Phase B features — spatial_recurrence_iou and same_zone_cross_page_fraction.

Intent: recurrence in previously approved zones must lower regression
suspicion (known dynamic areas), and cross-page failures in the same relative
zone signal a shared-component redesign. Missing evidence must stay NaN —
encoding it as 0 would fabricate a "no overlap" observation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from visualq_vrai.features.extract import _bbox_iou, featurize_bundle
from visualq_vrai.schema.bundle import (
    BBox,
    DiffBundle,
    DiffResultRow,
    HistoricalRunSummary,
    RunContext,
)


def _diff_row(**overrides) -> DiffResultRow:
    base = dict(
        scenario="home",
        viewport="desktop",
        browser="chromium",
        status="fail",
        misMatchPercentage=3.0,
        diffPixels=30000.0,
        totalPixels=1152000.0,
        dimensions={"width": 1280.0, "height": 900.0},
        diffBBox=BBox(x=100, y=200, width=400, height=300),
    )
    base.update(overrides)
    return DiffResultRow(**base)


def _bundle(**overrides) -> DiffBundle:
    base = dict(
        sampleId="s1",
        diff=_diff_row(),
        runContext=RunContext(
            runId="run-1",
            createdAt=datetime(2026, 7, 1, tzinfo=timezone.utc),
        ),
    )
    base.update(overrides)
    return DiffBundle(**base)


def _history(verdict: str | None, bbox: BBox | None, day: int) -> HistoricalRunSummary:
    return HistoricalRunSummary(
        runId=f"run-h{day}",
        createdAt=datetime(2026, 6, day, tzinfo=timezone.utc),
        misMatchPercentage=2.0,
        status="fail",
        verdict=verdict,
        diffBBox=bbox,
    )


def test_bbox_iou_exact():
    # Identical boxes → 1 ; disjoint → 0 ; half-overlapping → known value.
    a = (0.0, 0.0, 100.0, 100.0)
    assert _bbox_iou(a, a) == 1.0
    assert _bbox_iou(a, (200.0, 200.0, 50.0, 50.0)) == 0.0
    # b overlaps the right half of a: intersection 50*100, union 15000.
    b = (50.0, 0.0, 100.0, 100.0)
    assert math.isclose(_bbox_iou(a, b), 5000.0 / 15000.0)


def test_recurrence_iou_uses_only_approved_history():
    same_zone = BBox(x=100, y=200, width=400, height=300)
    other_zone = BBox(x=900, y=700, width=100, height=100)
    bundle = _bundle(
        history=[
            _history("intentional_redesign", same_zone, 1),
            _history("accidental_regression", same_zone, 2),
            _history("intentional_redesign", other_zone, 3),
        ]
    )
    row = featurize_bundle(bundle)
    # Max IoU over *approved* bboxes: exact match with the first entry.
    assert row["spatial_recurrence_iou"] == 1.0

    # Only a regression verdict in the same zone → not "approved recurrence".
    bundle2 = _bundle(history=[_history("accidental_regression", same_zone, 1)])
    assert math.isnan(featurize_bundle(bundle2)["spatial_recurrence_iou"])


def test_recurrence_iou_partial_overlap_exact_value():
    # Approved bbox shifted right by half the width: IoU = 200*300 / 600*300.
    approved = BBox(x=300, y=200, width=400, height=300)
    bundle = _bundle(history=[_history("intentional_redesign", approved, 1)])
    row = featurize_bundle(bundle)
    assert math.isclose(row["spatial_recurrence_iou"], (200 * 300) / (600 * 300))


def test_recurrence_nan_without_bbox_evidence():
    # No current bbox (no diffBBox, no diff image) → NaN, never 0.
    bundle = _bundle(
        diff=_diff_row(diffBBox=None),
        history=[_history("intentional_redesign", BBox(x=0, y=0, width=10, height=10), 1)],
    )
    assert math.isnan(featurize_bundle(bundle)["spatial_recurrence_iou"])
    # Approved history without bboxes → NaN as well.
    bundle2 = _bundle(history=[_history("intentional_redesign", None, 1)])
    assert math.isnan(featurize_bundle(bundle2)["spatial_recurrence_iou"])


def test_same_zone_cross_page_fraction():
    # Current diff at rel (100/1280, 200/900, 400/1280, 300/900).
    same_zone_sibling = _diff_row(
        scenario="about",
        diffBBox=BBox(x=100, y=200, width=400, height=300),
    )
    other_zone_sibling = _diff_row(
        scenario="contact",
        diffBBox=BBox(x=1000, y=750, width=50, height=50),
    )
    passing_sibling = _diff_row(scenario="pricing", status="pass", diffBBox=None)
    same_viewport_sibling = _diff_row(scenario="home")  # same scenario — excluded

    bundle = _bundle(
        runContext=RunContext(
            runId="run-1",
            createdAt=datetime(2026, 7, 1, tzinfo=timezone.utc),
            siblingResults=[
                same_zone_sibling,
                other_zone_sibling,
                passing_sibling,
                same_viewport_sibling,
            ],
        )
    )
    row = featurize_bundle(bundle)
    # 3 other-scenario siblings, 1 failing in the same relative zone.
    assert math.isclose(row["same_zone_cross_page_fraction"], 1.0 / 3.0)


def test_same_zone_scale_invariant_across_page_sizes():
    # Sibling page is 2x the size but the diff covers the same relative zone.
    scaled_sibling = _diff_row(
        scenario="about",
        dimensions={"width": 2560.0, "height": 1800.0},
        diffBBox=BBox(x=200, y=400, width=800, height=600),
    )
    bundle = _bundle(
        runContext=RunContext(
            runId="run-1",
            createdAt=datetime(2026, 7, 1, tzinfo=timezone.utc),
            siblingResults=[scaled_sibling],
        )
    )
    row = featurize_bundle(bundle)
    assert row["same_zone_cross_page_fraction"] == 1.0


def test_same_zone_nan_without_cross_page_evidence():
    # No other-scenario siblings → NaN.
    bundle = _bundle(
        runContext=RunContext(
            runId="run-1",
            createdAt=datetime(2026, 7, 1, tzinfo=timezone.utc),
            siblingResults=[_diff_row(scenario="home")],
        )
    )
    assert math.isnan(featurize_bundle(bundle)["same_zone_cross_page_fraction"])
