# Contributing to VisualQ VRAI

Thank you for contributing. This guide gets you from clone to a passing test suite in under 15 minutes.

**VRAI** = Visual Regression Analysis Intelligence — a 3-class diff triage engine for VRT pipelines (VisualQ-first, pipeline-agnostic).

---

## Table of contents

1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [Architecture](#architecture)
4. [Core concepts](#core-concepts)
5. [Development workflow](#development-workflow)
6. [Testing](#testing)
7. [Working with fixtures](#working-with-fixtures)
8. [HTTP service](#http-service)
9. [TabICL and feature constraints](#tabicl-and-feature-constraints)
10. [VisualQ bridge (private)](#visualq-bridge-private)
11. [Design principles](#design-principles)
12. [Pull request checklist](#pull-request-checklist)

---

## Quick start

```bash
git clone https://github.com/abecms/visualq-vrai.git
cd visualq-vrai

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Fast tests (~30s)
pytest -m "not slow"

# CLI smoke test
visualq-vrai fixtures/bundles

# Optional: full TabICL integration (~2 min)
pytest tests/test_tabicl_integration.py
```

**Requirements:** Python 3.11+. TabICL pulls PyTorch — first install may take several minutes.

---

## Repository layout

```
visualq-vrai/
├── src/visualq_vrai/
│   ├── schema/          # DiffBundle Pydantic models + feature column spec
│   ├── features/        # Featurization (G1–G12) + spatial diff analysis
│   ├── heuristic/       # Rule-based baseline (Phase 0)
│   ├── service/         # FastAPI app, TabICL predict, conformal selector
│   ├── eval/            # Evaluation harness (temporal + group splits)
│   └── cli/             # `visualq-vrai` demo command
├── tests/               # pytest suite
├── fixtures/
│   ├── bundles/         # DiffBundle JSON samples (synthetic + exported)
│   ├── images/          # Diff PNGs referenced by bundles
│   └── generate_synthetic.py
├── eval/run_eval.py     # Standalone eval script on fixtures
├── schema/              # Published JSON Schema (generated)
├── scripts/export_json_schema.py
├── pyproject.toml
├── README.md            # User-facing usage guide
└── CONTRIBUTING.md      # This file
```

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           DiffBundle JSON            │
                    │  (report row + images + metadata)    │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   featurize_bundle()  (~86 cols)    │
                    │   G1 magnitude … G12 AI distilled   │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   ┌──────────────────────┐                    ┌──────────────────────────┐
   │  Heuristic triage    │                    │  TabICL + conformal      │
   │  (always available)  │                    │  (needs ≥300 labels)     │
   └──────────────────────┘                    └────────────┬─────────────┘
              │                                              │
              │                              ┌───────────────┴───────────────┐
              │                              ▼                               ▼
              │                    predict_proba + SHAP          conformal autoTriage
              └──────────────────────────────┬───────────────────────────────┘
                                             ▼
                                    TriageResult per diff
```

**Two-stage product logic (not a fallback chain):**

- Heuristic = explicit cold-start floor; response `source: "heuristic"`.
- TabICL = model path; response `source: "tabicl"`.
- If context is insufficient, TabICL returns `insufficientData: true` and **does not** silently delegate to the LLM or heuristics inside `/predict`.

---

## Core concepts

### DiffBundle (`bundleVersion: 1`)

Defined in `src/visualq_vrai/schema/bundle.py`. Mirrors VisualQ's worker `DiffResult` row plus:

- `images` — paths/URLs to baseline, test, diff PNGs and DOM maps
- `runContext` — run metadata, CI signals, sibling results in the same run
- `intentSignals` — optional GitHub / JIRA / Figma blocks
- `history` — prior runs for as-of historical features (G7)
- `label` — human verdict for supervised context

Regenerate JSON Schema after schema changes:

```bash
python scripts/export_json_schema.py
# → schema/diff-bundle.v1.json
```

### Feature matrix (`schemaVersion: 1`)

86 columns in `src/visualq_vrai/schema/feature_spec.py`:

| Group | Count | Source |
|-------|-------|--------|
| G1 Magnitude | 6 | mismatch %, pixel ratios, page geometry |
| G2 Spatial | 12 | connected components on diff PNG |
| G3 Shift | 5 | layout shift regions |
| G4 DOM | 14 | elementResults / blockResults |
| G5 Config | 5 | scenario thresholds and rules |
| G6 Execution | 10 | viewport, browser, CI, time-of-day |
| G7 History | 10 | as-of prior runs (NaN if < 3) |
| G8 Cross-view | 5 | same-run viewport/browser consistency |
| G9 GitHub | 7 | intent signals (NaN if absent) |
| G10 JIRA | 4 | intent signals (NaN if absent) |
| G11 Figma | 4 | design alignment (NaN if absent) |
| G12 AI | 4 | Smart Diff distilled features |

**Missing data = `NaN`, never sentinel `0` or `-1`.** TabICL imputes NaN natively; fake zeros would leak false signal.

### Triage classes

```python
"intentional_redesign"
"accidental_regression"
"platform_constraint"
```

Label column in Parquet: `label_class`.

### Guardrails

| Constant | Value | Meaning |
|----------|-------|---------|
| `MIN_CONTEXT_ROWS` | 300 | Minimum labeled rows before TabICL serves |
| `MIN_CLASS_EXAMPLES` | 15 | Minimum per class in context |
| Conformal nominal risk | 2% | Max regression miss rate on auto-triaged subset |

---

## Development workflow

1. **Branch** from `main`: `feature/short-description`
2. **Change** the smallest surface that encodes the intent — no drive-by refactors
3. **Test** — add or update tests that fail if business logic regresses
4. **Schema** — if `DiffBundle` or feature columns change, bump `schemaVersion` / `bundleVersion` and fail loud on mismatch (never coerce)
5. **PR** — fill checklist below

### Editable install

Always develop with:

```bash
pip install -e ".[dev]"
```

After moving the repo directory, recreate `.venv` (paths are absolute in the venv).

---

## Testing

```bash
# Fast unit tests (recommended on every commit)
pytest -m "not slow"

# Everything including TabICL end-to-end (~2–3 min)
pytest

# Single module
pytest tests/test_features.py -v

# Eval harness on fixtures
python eval/run_eval.py
```

### Test markers

| Marker | Purpose |
|--------|---------|
| `slow` | TabICL fit/predict integration — deselect with `-m "not slow"` |

### What tests must encode

- **Intent**, not just happy path — e.g. absent connecteurs → NaN, not 0
- **Anti-leakage** for historical features (as-of correctness)
- **MIN_CONTEXT** gate returns `insufficientData`, not a forced class
- **schemaVersion mismatch** → 422 on API, ValueError in library

---

## Working with fixtures

### Synthetic set (committed)

```bash
python fixtures/generate_synthetic.py
# Writes 360 bundles to fixtures/bundles/ + PNGs to fixtures/images/
```

Patterns map to classes: `wide` → redesign, `block` → regression, `noise` → platform.

### Real VisualQ data (private export)

From the VisualQ monorepo sibling:

```bash
cd visualq
npx tsx scripts/export-diff-bundles.ts \
  --org ORG_ID \
  --project PROJECT_ID \
  --out ../visualq-vrai/fixtures/bundles \
  --limit 100
```

Export anonymizes IDs (SHA-256), strips client names, backfills `intentional_redesign` from past approvals.

**Do not commit** exports containing identifiable customer data.

---

## HTTP service

```bash
uvicorn visualq_vrai.service.app:app --reload --port 8090
```

Interactive docs: http://localhost:8090/docs

| File | Role |
|------|------|
| `service/app.py` | FastAPI routes |
| `service/predict.py` | TabICL fit/predict, Parquet codec, SHAP via `tabicl.shap` |
| `service/conformal.py` | Split conformal deferral |

### Parquet contract

Requests carry base64 Parquet with columns from `FEATURE_COLUMNS` plus metadata columns (`sampleId`, `runId`, `createdAt`, `label_class`, …).

Use `encode_parquet()` / `_decode_parquet()` from `service/predict.py`.

### Active columns

TabICL drops all-NaN columns internally. `predict.py` computes `_active_columns()` so fit and predict use the **same** column set — do not bypass this.

---

## TabICL and feature constraints

From TabICLv2 docs (encoded in our design):

1. **~2–100 columns, 300–60K rows** for in-context learning
2. **NaN = missing** — never impute with 0 in the featurizer
3. **Outliers** — use `log1p` / ratios (see G1, G2)
4. **No identity columns** — no `scenarioId` in X; history carries recurrence signal
5. **schemaVersion** must match between context and query

SHAP:

```python
from tabicl.shap import get_shap_values
sv = get_shap_values(fitted_model, x_query, attribute_names=active_cols)
```

---

## VisualQ bridge (private)

VisualQ lives in a separate repo/directory. Integration path:

1. Worker produces `report.json` + PNGs + DOM maps (unchanged)
2. VisualQ exports or builds `DiffBundle` at run completion
3. VisualQ calls `POST /predict` on the VRAI service
4. Results merged into viewer as `triage` field

This repo intentionally has **no** Firebase/AWS SDK. The only private bridge today is `visualq/scripts/export-diff-bundles.ts`.

Post-hackathon production target: ECS Fargate service `visualq-ml` (Terraform in VisualQ `infrastructure/`).

---

## Design principles

These are non-negotiable for this codebase:

1. **Fail loud** — wrong schema, insufficient context, or missing binding → explicit error or `insufficientData`, never silent substitution
2. **No repair fallbacks** — do not paper over bad inputs with secondary code paths; fix the producer
3. **One path per step** — heuristic and TabICL are separate products surfaces, not try-A-then-B
4. **Absent connector = NaN block** — GitHub/JIRA/Figma columns stay NaN when signals are missing
5. **Conformal before auto-action** — no auto-approve/reject unless `conformal.autoTriage === true`

---

## Pull request checklist

- [ ] Tests pass: `pytest -m "not slow"`
- [ ] New behavior has tests explaining **why** it matters
- [ ] If `DiffBundle` changed: `python scripts/export_json_schema.py` + bump version
- [ ] If feature columns changed: update `FEATURE_COLUMNS`, `schemaVersion`, and tests
- [ ] README updated for user-visible changes
- [ ] No secrets, customer names, or internal URLs in fixtures
- [ ] PR description states heuristic vs TabICL impact and cold-start behavior

---

## Getting help

- User usage → [README.md](README.md)
- JSON Schema → [schema/diff-bundle.v1.json](schema/diff-bundle.v1.json)
- VisualQ VRT types (reference) → VisualQ worker `DiffResult` in `worker/src/types.ts`

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
