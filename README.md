# VisualQ VRAI

**V**isual **R**egression **A**nalysis **I**ntelligence — open-source triage for visual regression test (VRT) diffs.

When a screenshot comparison fails, VRAI helps you decide *what kind of failure it is* and what to do next — without reading every pixel manually.

## What it does

For each VRT diff, VRAI assigns one of three classes:

| Class | Meaning | Typical action |
|-------|---------|----------------|
| `intentional_redesign` | Deliberate visual change (new design, expected content) | Approve / update baseline |
| `accidental_regression` | Unintended visual bug introduced by code | Investigate and fix |
| `platform_constraint` | Environment noise (browser rendering, anti-aliasing, third-party widgets) | Add a comparison rule — not a code fix |

VRAI combines:

1. **Heuristic rules** — works from day one, no training data required.
2. **TabICL model** — learns from your project's labeled history (in-context learning, no retraining pipeline).
3. **Conformal gate** — auto-triage only when the statistical risk of missing a regression is bounded.
4. **SHAP explanations** — top features driving each prediction.

## Who is this for?

- **VisualQ users** — triage diffs exported from VisualQ runs (primary integration path).
- **Any VRT pipeline** — as long as you can produce a [DiffBundle](#input-diffbundle) JSON file per diff (screenshot row + optional DOM maps and metadata).

VRAI has **no dependency** on Firebase, Firestore, or AWS. It runs locally or as a standalone HTTP service.

---

## Installation

**Requirements:** Python 3.11+, ~2 GB disk for TabICL dependencies.

```bash
git clone https://github.com/abecms/visualq-vrai.git
cd visualq-vrai
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

For development and tests, install dev extras:

```bash
pip install -e ".[dev]"
```

---

## Quick start (CLI)

Bundled synthetic fixtures are included for a smoke test:

```bash
visualq-vrai fixtures/bundles
```

Predict a specific sample:

```bash
visualq-vrai fixtures/bundles --query synthetic-0355
```

**Output:**

- Heuristic triage (always shown).
- TabICL triage + probabilities + SHAP features (when ≥ 300 labeled samples exist in the fixture set).

---

## Quick start (HTTP API)

Start the service:

```bash
uvicorn visualq_vrai.service.app:app --host 0.0.0.0 --port 8090
```

Health check:

```bash
curl http://localhost:8090/health
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `POST` | `/predict` | TabICL triage (+ conformal gate, SHAP) |
| `POST` | `/heuristic` | Rule-based triage only (cold start) |

Both `POST` endpoints accept **base64-encoded Parquet** tables (see [Preparing data](#preparing-data-from-diffbundles)).

### Example: heuristic triage

```python
import base64
import io
import json
from pathlib import Path

import httpx
import pandas as pd

from visualq_vrai.features.extract import bundles_to_dataframe
from visualq_vrai.schema.bundle import DiffBundle

# Load your DiffBundles
bundles = [
    DiffBundle.model_validate(json.loads(p.read_text()))
    for p in Path("fixtures/bundles").glob("*.json")
]
df = bundles_to_dataframe(bundles)
query = df.iloc[[-1]]

buf = io.BytesIO()
query.to_parquet(buf, index=False)
payload = {
    "schemaVersion": 1,
    "contextParquet": "",  # not used by /heuristic
    "queryParquet": base64.b64encode(buf.getvalue()).decode("ascii"),
}

resp = httpx.post("http://localhost:8090/heuristic", json=payload, timeout=60)
print(resp.json())
```

### Example response

```json
{
  "schemaVersion": 1,
  "results": [
    {
      "class": "accidental_regression",
      "probabilities": {
        "intentional_redesign": 0.0,
        "accidental_regression": 0.7,
        "platform_constraint": 0.0
      },
      "topFeatures": [
        { "feature": "interactive_elem_changed", "shap": 0.0 }
      ],
      "conformal": { "autoTriage": false, "riskLevel": null },
      "source": "heuristic"
    }
  ]
}
```

TabICL `/predict` responses add calibrated probabilities, SHAP values, and a conformal `autoTriage` flag. When training context is insufficient:

```json
{
  "insufficientData": true,
  "insufficientReason": "context_rows_below_300",
  "source": "heuristic"
}
```

---

## Input: DiffBundle

Each diff is one **DiffBundle** JSON file (`bundleVersion: 1`).

Minimal shape:

```json
{
  "bundleVersion": 1,
  "sampleId": "unique-id",
  "diff": {
    "scenario": "home",
    "viewport": "desktop",
    "browser": "chromium",
    "status": "fail",
    "misMatchPercentage": 2.4,
    "diffPixels": 12000,
    "totalPixels": 1152000,
    "dimensions": { "width": 1280, "height": 900 }
  },
  "images": {
    "diffPng": "fixtures/images/my_diff.png"
  },
  "runContext": {
    "runId": "run-abc",
    "createdAt": "2026-07-04T10:00:00Z"
  }
}
```

Optional but valuable fields:

- `images.baselinePng`, `images.testPng`, DOM map paths
- `diff.elementResults`, `diff.blockResults`, `diff.shiftRegions`
- `runContext.ci` — commit, branch, PR, JIRA key
- `runContext.siblingResults` — other diffs in the same run
- `intentSignals` — GitHub / JIRA / Figma intent (absent block → NaN features, never `0`)
- `label.verdict` — human label for training context (`intentional_redesign`, `accidental_regression`, `platform_constraint`)

Full schema: [`schema/diff-bundle.v1.json`](schema/diff-bundle.v1.json)

---

## Preparing data from DiffBundles

```python
from visualq_vrai.features.extract import bundles_to_dataframe, featurize_bundle
from visualq_vrai.service.predict import encode_parquet

# One row per diff, ~86 numeric features
df = bundles_to_dataframe(bundles)

# Labeled rows → context (training)
context_df = df.dropna(subset=["label_class"])

# Row(s) to predict → query
query_df = df.iloc[[42]]

context_b64 = encode_parquet(context_df)
query_b64 = encode_parquet(query_df)
```

Send both to `POST /predict`.

---

## VisualQ integration

If you use [VisualQ](https://visualq.ai), export anonymized bundles from your private VisualQ workspace:

```bash
cd visualq
npx tsx scripts/export-diff-bundles.ts \
  --org YOUR_ORG \
  --project YOUR_PROJECT \
  --out ../visualq-vrai/fixtures/bundles
```

Then run VRAI locally or deploy the HTTP service and point VisualQ at it (production path: ECS Fargate `visualq-ml`).

---

## Cold start and auto-triage

| Stage | Labeled diffs | What you get |
|-------|---------------|--------------|
| Day 1 | 0 | Heuristic triage only |
| Growing | 1–299 | Heuristic + TabICL deferred (`insufficientData`) |
| Production | ≥ 300 (≥ 15 per class) | TabICL probabilities + SHAP + conformal auto-triage |

The conformal selector defers uncertain cases to a human rather than forcing a class. **Never treat heuristic output as a silent fallback** — check the `source` field.

---

## When to use which endpoint

| Situation | Endpoint |
|-----------|----------|
| No labels yet / hackathon demo | `/heuristic` |
| Enough labeled history per project | `/predict` |
| Need explanations for reviewers | `/predict` (SHAP in `topFeatures`) |
| Automated CI gate | `/predict` only when `conformal.autoTriage === true` |

---

## Documentation for contributors

Architecture, feature dictionary, test workflow, and PR guidelines:

→ **[CONTRIBUTING.md](CONTRIBUTING.md)**

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
