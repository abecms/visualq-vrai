"""Tests for featurization."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vrt_triage.features.extract import featurize_bundle
from vrt_triage.schema.bundle import (
    BUNDLE_VERSION,
    BundleImages,
    DiffBundle,
    DiffBundleLabel,
    DiffResultRow,
    RunContext,
)
from vrt_triage.schema.feature_spec import FEATURE_COLUMNS, LABEL_COLUMN


def _bundle(**kwargs) -> DiffBundle:
    defaults = dict(
        bundleVersion=BUNDLE_VERSION,
        sampleId="s1",
        diff=DiffResultRow(
            scenario="home",
            viewport="desktop",
            browser="chromium",
            status="fail",
            misMatchPercentage=3.0,
            diffPixels=300,
            totalPixels=10000,
            dimensions={"width": 200, "height": 100},
        ),
        runContext=RunContext(
            runId="r1",
            createdAt=datetime(2026, 3, 1, tzinfo=timezone.utc),
        ),
    )
    defaults.update(kwargs)
    return DiffBundle(**defaults)


def test_feature_vector_has_all_columns():
    row = featurize_bundle(_bundle())
    for col in FEATURE_COLUMNS:
        assert col in row


def test_intent_absent_is_nan_not_zero():
    row = featurize_bundle(_bundle(intentSignals=None))
    assert np.isnan(row["commit_changed_since_baseline"])
    assert np.isnan(row["jira_key_present"])
    assert np.isnan(row["figma_alignment_delta"])


def test_spatial_from_diff_png(tmp_path):
    from PIL import Image, ImageDraw

    diff = tmp_path / "diff.png"
    img = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 10, 40, 30), fill=(255, 0, 0, 255))
    img.save(diff)

    bundle = _bundle(images=BundleImages(diffPng=str(diff)))
    row = featurize_bundle(bundle)
    assert row["region_count_log"] > 0
    assert row["largest_region_area_ratio"] > 0


def test_label_propagation():
    bundle = _bundle(
        label=DiffBundleLabel(verdict="accidental_regression", verdictSource="viewer"),
    )
    row = featurize_bundle(bundle)
    assert row[LABEL_COLUMN] == "accidental_regression"
