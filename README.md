# vrt-triage

Open-source **visual regression diff triage** for VisualQ and compatible VRT pipelines.

Classifies each VRT diff into three classes:

- `intentional_redesign` — deliberate visual change (approve baseline)
- `accidental_regression` — unintended visual bug
- `platform_constraint` — environment/rendering noise (comparison rule, not a code fix)

## Architecture

```
DiffBundle (JSON) → featurize (~86 columns) → heuristic baseline OR TabICL + conformal + SHAP
```

Consumes **DiffBundle** artefacts exported from VisualQ (`report.json` row + DOM maps + images + optional intent signals). No Firebase, Firestore, or AWS dependency in this repo.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Demo on bundled fixtures
vrt-triage fixtures/bundles

# Start prediction API
uvicorn vrt_triage.service.app:app --reload --port 8090
```

## DiffBundle contract

See [`schema/diff-bundle.v1.json`](schema/diff-bundle.v1.json) and [`src/vrt_triage/schema/bundle.py`](src/vrt_triage/schema/bundle.py).

Key fields:

- `bundleVersion: 1`
- `diff` — one `DiffResult` row (mismatch %, shift regions, element/block results)
- `images` — local paths or URLs for baseline/test/diff PNGs and DOM maps
- `runContext` — viewport, browser, CI metadata, sibling results in the same run
- `intentSignals` — optional GitHub/JIRA/Figma signals (`null` block → NaN features)
- `label` — optional human verdict for training/eval

## VisualQ integration (private)

VisualQ exports anonymized bundles via:

```bash
cd visualq
npx tsx scripts/export-diff-bundles.ts --org ORG --project PROJECT --out ../vrt-triage/fixtures/bundles
```

After the hackathon, VisualQ calls `POST /predict` with live bundles (ECS Fargate `visualq-ml`).

## API

### `POST /predict`

```json
{
  "schemaVersion": 1,
  "contextParquet": "<base64 parquet>",
  "queryParquet": "<base64 parquet>"
}
```

Returns probabilities, SHAP top features, conformal auto-triage gate, or `{ "insufficientData": true }` when labeled context &lt; 300 rows or &lt; 15 per class.

### `POST /heuristic`

Same Parquet inputs; always returns rule-based triage (cold-start floor).

## License

Apache-2.0 — see [LICENSE](LICENSE).
