"""Generate synthetic DiffBundle fixtures with class-controlled signatures.

Each class gets the signature the real pipeline would produce:

- ``platform_constraint`` — scattered isolated pixels, often mono-browser,
  low mismatch vs history, recurring in previously approved zones, no code
  change since baseline.
- ``intentional_redesign`` — large coherent regions, broad run failure,
  design-flavored intent signals (feat/style PR, design label, Figma delta
  moving toward the target).
- ``accidental_regression`` — concentrated region on an interactive element,
  high mismatch z-score vs history, UI code changed, Figma delta moving away.

Signals are noisy and partially missing on purpose (connector coverage,
flaky heuristic cues) so the heuristic floor is beaten but not saturated,
leaving multivariate structure for the model to exploit.

Reproducible: seeded RNG, regenerate with ``python fixtures/generate_synthetic.py``.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from visualq_vrai.schema.bundle import (
    BUNDLE_VERSION,
    BBox,
    BundleImages,
    CiContext,
    DiffBundle,
    DiffBundleLabel,
    DiffResultRow,
    ElementDiffResult,
    FigmaIntentSignals,
    GitHubIntentSignals,
    HistoricalRunSummary,
    IntentSignals,
    JiraIntentSignals,
    RunContext,
)

WIDTH, HEIGHT = 1280, 900
SCENARIOS = ["home", "about", "pricing", "blog", "contact"]
CLASSES = ["intentional_redesign", "accidental_regression", "platform_constraint"]

# Approved dynamic zone reused by platform-constraint history (e.g. a carousel).
RECURRING_ZONE = BBox(x=200, y=300, width=500, height=250)


def _write_diff_png(path: Path, pattern: str, rng: random.Random) -> BBox:
    """Draw the diff PNG and return the union bbox of drawn pixels."""
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if pattern == "noise":
        xs, ys = [], []
        for _ in range(rng.randint(200, 900)):
            x, y = rng.randint(0, WIDTH - 1), rng.randint(0, HEIGHT - 1)
            xs.append(x)
            ys.append(y)
            draw.point((x, y), fill=(255, 0, 0, 255))
        bbox = BBox(x=min(xs), y=min(ys), width=max(xs) - min(xs) + 1, height=max(ys) - min(ys) + 1)
    elif pattern == "block":
        x0 = rng.randint(50, WIDTH // 2)
        y0 = rng.randint(50, HEIGHT // 2)
        w = rng.randint(WIDTH // 8, WIDTH // 3)
        h = rng.randint(HEIGHT // 10, HEIGHT // 4)
        draw.rectangle((x0, y0, x0 + w, y0 + h), fill=(255, 0, 0, 255))
        bbox = BBox(x=x0, y=y0, width=w + 1, height=h + 1)
    elif pattern == "recurring":
        # Diff inside the historically approved dynamic zone.
        z = RECURRING_ZONE
        jitter = rng.randint(-20, 20)
        draw.rectangle(
            (z.x + jitter, z.y + jitter, z.x + z.width + jitter, z.y + z.height + jitter),
            fill=(255, 0, 0, 255),
        )
        bbox = BBox(x=z.x + jitter, y=z.y + jitter, width=z.width + 1, height=z.height + 1)
    else:  # "wide" — full-width band, shared-component redesign
        y0 = rng.choice([0, HEIGHT // 3, 2 * HEIGHT // 3])
        h = HEIGHT // 4
        draw.rectangle((0, y0, WIDTH, y0 + h), fill=(255, 0, 0, 255))
        bbox = BBox(x=0, y=y0, width=WIDTH, height=h + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return bbox


def _history(
    verdict_class: str,
    current_mismatch: float,
    base_time: datetime,
    rng: random.Random,
) -> list[HistoricalRunSummary]:
    """~10 prior runs whose stats produce the intended z-score regime."""
    entries: list[HistoricalRunSummary] = []
    if verdict_class == "accidental_regression":
        # Quiet history → current mismatch is a spike (high z-score).
        mismatches = [rng.uniform(0.01, 0.3) for _ in range(10)]
        statuses = ["pass"] * 8 + ["fail", "pass"]
        verdicts: list[str | None] = [None] * 10
        bboxes: list[BBox | None] = [None] * 10
    elif verdict_class == "platform_constraint":
        # Noisy history: recurrent small fails, some approved in the same zone.
        mismatches = [rng.uniform(0.1, current_mismatch * 1.5) for _ in range(10)]
        statuses = [rng.choice(["fail", "pass", "fail"]) for _ in range(10)]
        verdicts = [
            "intentional_redesign" if rng.random() < 0.4 else None for _ in range(10)
        ]
        bboxes = [
            RECURRING_ZONE if v == "intentional_redesign" else None for v in verdicts
        ]
    else:  # intentional_redesign
        mismatches = [rng.uniform(0.05, 1.0) for _ in range(10)]
        statuses = ["pass"] * 7 + ["fail"] * 3
        verdicts = [None] * 10
        bboxes = [None] * 10
    rng.shuffle(statuses)
    for i in range(10):
        entries.append(
            HistoricalRunSummary(
                runId=f"hist-{i}",
                createdAt=base_time - timedelta(days=10 - i),
                misMatchPercentage=mismatches[i],
                status=statuses[i],
                verdict=verdicts[i],
                diffBBox=bboxes[i],
            )
        )
    return entries


def _siblings(
    verdict_class: str,
    scenario: str,
    mismatch: float,
    diff_bbox: BBox,
    rng: random.Random,
) -> list[DiffResultRow]:
    """Same-run results across browsers and other scenarios."""
    rows: list[DiffResultRow] = []
    mono_browser = verdict_class == "platform_constraint" and rng.random() < 0.7
    broad_failure = verdict_class == "intentional_redesign" and rng.random() < 0.8

    for browser in ("chromium", "firefox", "webkit"):
        if browser == "chromium":
            continue  # the bundle's own row
        fails = not mono_browser and (broad_failure or rng.random() < 0.3)
        rows.append(
            DiffResultRow(
                scenario=scenario,
                viewport="desktop",
                browser=browser,
                status="fail" if fails else "pass",
                misMatchPercentage=mismatch * rng.uniform(0.6, 1.1) if fails else 0.0,
                diffPixels=1000.0 if fails else 0.0,
                totalPixels=float(WIDTH * HEIGHT),
                dimensions={"width": float(WIDTH), "height": float(HEIGHT)},
            )
        )

    for other in SCENARIOS:
        if other == scenario:
            continue
        # Shared-component redesign hits the same relative zone on other pages.
        same_zone_fail = broad_failure and rng.random() < 0.8
        random_fail = not same_zone_fail and rng.random() < 0.15
        fails = same_zone_fail or random_fail
        bbox = None
        if same_zone_fail:
            bbox = BBox(
                x=diff_bbox.x, y=diff_bbox.y,
                width=diff_bbox.width, height=diff_bbox.height,
            )
        elif random_fail:
            bbox = BBox(x=rng.randint(0, 800), y=rng.randint(0, 600), width=120, height=90)
        rows.append(
            DiffResultRow(
                scenario=other,
                viewport="desktop",
                browser="chromium",
                status="fail" if fails else "pass",
                misMatchPercentage=mismatch * rng.uniform(0.5, 1.2) if fails else 0.0,
                diffPixels=1000.0 if fails else 0.0,
                totalPixels=float(WIDTH * HEIGHT),
                dimensions={"width": float(WIDTH), "height": float(HEIGHT)},
                diffBBox=bbox,
            )
        )
    return rows


def _element_results(verdict_class: str, diff_bbox: BBox, rng: random.Random) -> list[ElementDiffResult] | None:
    if rng.random() < 0.25:
        return None  # DOM map missing — G4 must stay NaN
    elements: list[ElementDiffResult] = []
    if verdict_class == "accidental_regression":
        elements.append(
            ElementDiffResult(
                selector="main > form button.submit",
                tag="button",
                bbox=diff_bbox,
                status="visual_change",
                mode="strict",
                diffPercentage=rng.uniform(20, 80),
                matchConfidence=rng.uniform(0.8, 1.0),
            )
        )
    elif verdict_class == "intentional_redesign":
        elements.append(
            ElementDiffResult(
                selector="header nav",
                tag="nav",
                semanticLabel="main navigation",
                bbox=diff_bbox,
                status="visual_change",
                mode="strict",
                diffPercentage=rng.uniform(30, 90),
                matchConfidence=rng.uniform(0.85, 1.0),
            )
        )
    else:
        elements.append(
            ElementDiffResult(
                selector="section.carousel",
                tag="div",
                semanticLabel="carousel",
                bbox=diff_bbox,
                status="dynamic",
                mode="layout",
                diffPixelCount=rng.uniform(100, 500),
                diffPercentage=rng.uniform(1, 8),
                matchConfidence=rng.uniform(0.7, 0.95),
            )
        )
    elements.append(
        ElementDiffResult(
            selector="footer",
            tag="footer",
            bbox=BBox(x=0, y=HEIGHT - 80, width=WIDTH, height=80),
            status="unchanged",
            mode="strict",
            matchConfidence=1.0,
        )
    )
    return elements


def _intent_signals(verdict_class: str, rng: random.Random) -> IntentSignals | None:
    # ~35% of bundles have no connectors configured at all (coverage strata).
    if rng.random() < 0.35:
        return None
    if verdict_class == "intentional_redesign":
        github = GitHubIntentSignals(
            commitChangedSinceBaseline=True,
            prFilesChanged=rng.randint(8, 40),
            uiFilesChanged=True,
            cssLinesChanged=rng.randint(100, 900),
            prTitleType=rng.choice(["feat", "style", "feat"]),
            prHasDesignLabel=rng.random() < 0.6,
            changedFilesMatchPage=rng.random() < 0.7,
        )
        jira = JiraIntentSignals(
            jiraKeyPresent=True,
            jiraIssueIsStory=rng.random() < 0.7,
            jiraIssueIsBug=False,
            jiraHasDesignLabel=rng.random() < 0.5,
        )
        figma = FigmaIntentSignals(
            figmaDesignChangedRecently=rng.random() < 0.7,
            daysSinceFigmaVersion=rng.uniform(1, 20),
            figmaAlignmentBaseline=rng.uniform(0.5, 0.8),
            figmaAlignmentDelta=rng.uniform(0.02, 0.25),  # moving toward target
        )
    elif verdict_class == "accidental_regression":
        github = GitHubIntentSignals(
            commitChangedSinceBaseline=True,
            prFilesChanged=rng.randint(1, 15),
            uiFilesChanged=rng.random() < 0.8,
            cssLinesChanged=rng.randint(0, 120),
            prTitleType=rng.choice(["fix", "refactor", "feat", "chore"]),
            prHasDesignLabel=False,
            changedFilesMatchPage=rng.random() < 0.5,
        )
        jira = JiraIntentSignals(
            jiraKeyPresent=rng.random() < 0.6,
            jiraIssueIsStory=rng.random() < 0.2,
            jiraIssueIsBug=rng.random() < 0.5,
            jiraHasDesignLabel=False,
        )
        figma = FigmaIntentSignals(
            figmaDesignChangedRecently=rng.random() < 0.15,
            daysSinceFigmaVersion=rng.uniform(30, 200),
            figmaAlignmentBaseline=rng.uniform(0.6, 0.9),
            figmaAlignmentDelta=rng.uniform(-0.25, -0.02),  # moving away
        )
    else:  # platform_constraint — no code change since baseline
        github = GitHubIntentSignals(
            commitChangedSinceBaseline=rng.random() < 0.2,
            prFilesChanged=0,
            uiFilesChanged=False,
            cssLinesChanged=0,
            prTitleType="chore",
            prHasDesignLabel=False,
            changedFilesMatchPage=False,
        )
        jira = JiraIntentSignals(
            jiraKeyPresent=rng.random() < 0.2,
            jiraIssueIsStory=False,
            jiraIssueIsBug=rng.random() < 0.1,
            jiraHasDesignLabel=False,
        )
        figma = FigmaIntentSignals(
            figmaDesignChangedRecently=False,
            daysSinceFigmaVersion=rng.uniform(40, 300),
            figmaAlignmentBaseline=rng.uniform(0.6, 0.9),
            figmaAlignmentDelta=rng.uniform(-0.02, 0.02),  # unchanged
        )
    # Partial coverage: drop whole connector blocks independently.
    return IntentSignals(
        github=github if rng.random() < 0.8 else None,
        jira=jira if rng.random() < 0.6 else None,
        figma=figma if rng.random() < 0.4 else None,
    )


def generate_synthetic_bundle(
    *,
    sample_id: str,
    verdict: str | None,
    verdict_class: str,
    rng: random.Random,
    base_time: datetime,
    project_id: str = "demo-project",
) -> DiffBundle:
    repo_root = Path(__file__).resolve().parents[1]
    images_dir = repo_root / "fixtures" / "images"
    diff_path = images_dir / f"{sample_id}_diff.png"

    scenario = rng.choice(SCENARIOS)
    if verdict_class == "platform_constraint":
        pattern = "recurring" if rng.random() < 0.5 else "noise"
        mismatch = rng.uniform(0.05, 1.2)
    elif verdict_class == "accidental_regression":
        pattern = "block"
        mismatch = rng.uniform(1.0, 9.0)
    else:
        pattern = "wide"
        mismatch = rng.uniform(4.0, 25.0)

    diff_bbox = _write_diff_png(diff_path, pattern, rng)

    ci = None
    if rng.random() < 0.7:
        ci = CiContext(
            commitSha=f"{rng.getrandbits(64):016x}",
            branch="main" if rng.random() < 0.5 else f"feature/{rng.randint(100, 999)}",
            prNumber=rng.randint(1, 500) if rng.random() < 0.6 else None,
            isMainBranch=rng.random() < 0.5,
        )

    return DiffBundle(
        bundleVersion=BUNDLE_VERSION,
        sampleId=sample_id,
        projectId=project_id,
        diff=DiffResultRow(
            scenario=scenario,
            viewport="desktop",
            browser="chromium",
            status="fail",
            misMatchPercentage=mismatch,
            diffPixels=float(int(mismatch * WIDTH * HEIGHT / 100)),
            totalPixels=float(WIDTH * HEIGHT),
            dimensions={"width": float(WIDTH), "height": float(HEIGHT)},
            baselineHeight=float(HEIGHT),
            elementResults=_element_results(verdict_class, diff_bbox, rng),
            diffBBox=diff_bbox if rng.random() < 0.5 else None,
        ),
        images=BundleImages(diffPng=str(diff_path.relative_to(repo_root))),
        runContext=RunContext(
            runId=f"run-{sample_id}",
            createdAt=base_time,
            trigger="ci",
            envIsProd=True,
            ci=ci,
            siblingResults=_siblings(verdict_class, scenario, mismatch, diff_bbox, rng),
        ),
        intentSignals=_intent_signals(verdict_class, rng),
        history=_history(verdict_class, mismatch, base_time, rng),
        label=(
            DiffBundleLabel(verdict=verdict, verdictSource="synthetic")
            if verdict
            else None
        ),
    )


def generate_fixture_set(out_dir: Path, n: int = 360, n_unlabeled: int = 15) -> None:
    rng = random.Random(42)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.json"):
        stale.unlink()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        verdict_class = CLASSES[i % 3]
        bundle = generate_synthetic_bundle(
            sample_id=f"synthetic-{i:04d}",
            verdict=verdict_class if i < n - n_unlabeled else None,
            verdict_class=verdict_class,
            rng=rng,
            base_time=base + timedelta(hours=i),
        )
        path = out_dir / f"synthetic-{i:04d}.json"
        path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    generate_fixture_set(root / "fixtures" / "bundles", n=360)
    print("Regenerated 360 fixture bundles.")
