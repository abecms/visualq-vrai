"""Tests for DiffBundle schema."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from vrt_triage.schema.bundle import BUNDLE_VERSION, DiffBundle, DiffResultRow, RunContext


def test_invalid_bundle_version_rejected():
    with pytest.raises(ValidationError):
        DiffBundle.model_validate({
            "bundleVersion": 2,
            "sampleId": "x",
            "diff": {
                "scenario": "home",
                "viewport": "desktop",
                "browser": "chromium",
                "status": "fail",
                "misMatchPercentage": 1.0,
                "diffPixels": 10,
                "totalPixels": 1000,
                "dimensions": {"width": 100, "height": 100},
            },
            "runContext": {
                "runId": "r1",
                "createdAt": "2026-01-01T00:00:00Z",
            },
        })


def test_valid_minimal_bundle():
    bundle = DiffBundle(
        bundleVersion=BUNDLE_VERSION,
        sampleId="sample-1",
        diff=DiffResultRow(
            scenario="home",
            viewport="desktop",
            browser="chromium",
            status="fail",
            misMatchPercentage=2.5,
            diffPixels=100,
            totalPixels=10000,
            dimensions={"width": 1280, "height": 900},
        ),
        runContext=RunContext(
            runId="run-1",
            createdAt=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    )
    assert bundle.bundleVersion == 1
    data = json.loads(bundle.model_dump_json())
    assert data["sampleId"] == "sample-1"
