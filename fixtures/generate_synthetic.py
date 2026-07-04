"""Generate synthetic DiffBundle fixtures for tests and hackathon demo."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from vrt_triage.schema.bundle import (
    BUNDLE_VERSION,
    BundleImages,
    DiffBundle,
    DiffBundleLabel,
    DiffResultRow,
    RunContext,
)


def _write_diff_png(path: Path, width: int, height: int, pattern: str, rng: random.Random) -> None:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if pattern == "noise":
        for _ in range(rng.randint(200, 800)):
            x, y = rng.randint(0, width - 1), rng.randint(0, height - 1)
            draw.point((x, y), fill=(255, 0, 0, 255))
    elif pattern == "block":
        x0 = rng.randint(0, width // 4)
        y0 = rng.randint(0, height // 4)
        x1 = x0 + rng.randint(width // 8, width // 3)
        y1 = y0 + rng.randint(height // 8, height // 3)
        draw.rectangle((x0, y0, x1, y1), fill=(255, 0, 0, 255))
    elif pattern == "wide":
        draw.rectangle((0, height // 3, width, 2 * height // 3), fill=(255, 0, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def generate_synthetic_bundle(
    *,
    sample_id: str,
    verdict: str | None,
    pattern: str,
    mismatch: float,
    rng: random.Random,
    base_time: datetime,
    project_id: str = "demo-project",
) -> DiffBundle:
    width, height = 1280, 900
    repo_root = Path(__file__).resolve().parents[1]
    images_dir = repo_root / "fixtures" / "images"
    diff_path = images_dir / f"{sample_id}_diff.png"
    _write_diff_png(diff_path, width, height, pattern, rng)

    only_browser = pattern == "noise" and rng.random() > 0.5
    sibling = DiffResultRow(
        scenario="home",
        viewport="desktop",
        browser="firefox" if only_browser else "chromium",
        status="pass" if only_browser else "fail",
        misMatchPercentage=0.0 if only_browser else mismatch * 0.8,
        diffPixels=100,
        totalPixels=width * height,
        dimensions={"width": width, "height": height},
    )

    bundle = DiffBundle(
        bundleVersion=BUNDLE_VERSION,
        sampleId=sample_id,
        projectId=project_id,
        diff=DiffResultRow(
            scenario="home",
            viewport="desktop",
            browser="chromium",
            status="fail",
            misMatchPercentage=mismatch,
            diffPixels=int(mismatch * width * height / 100),
            totalPixels=width * height,
            dimensions={"width": width, "height": height},
            baselineHeight=float(height),
        ),
        images=BundleImages(
            diffPng=str(diff_path.relative_to(repo_root)),
        ),
        runContext=RunContext(
            runId=f"run-{sample_id}",
            createdAt=base_time,
            trigger="ci",
            envIsProd=True,
            siblingResults=[sibling],
        ),
        label=DiffBundleLabel(verdict=verdict, verdictSource="synthetic") if verdict else None,
    )
    return bundle


def generate_fixture_set(out_dir: Path, n: int = 360) -> None:
    rng = random.Random(42)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = [
        "intentional_redesign",
        "accidental_regression",
        "platform_constraint",
    ]
    patterns = {
        "intentional_redesign": "wide",
        "accidental_regression": "block",
        "platform_constraint": "noise",
    }
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        verdict = classes[i % 3]
        pattern = patterns[verdict]
        mismatch = {
            "intentional_redesign": rng.uniform(5, 25),
            "accidental_regression": rng.uniform(1, 8),
            "platform_constraint": rng.uniform(0.05, 0.8),
        }[verdict]
        bundle = generate_synthetic_bundle(
            sample_id=f"synthetic-{i:04d}",
            verdict=verdict if i < n - 5 else None,
            pattern=pattern,
            mismatch=mismatch,
            rng=rng,
            base_time=base + timedelta(hours=i),
        )
        path = out_dir / f"synthetic-{i:04d}.json"
        path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    generate_fixture_set(root / "fixtures" / "bundles", n=360)
