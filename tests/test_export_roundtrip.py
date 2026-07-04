"""Round-trip with the VisualQ exporter output.

Intent: `fixtures/export-samples/exported-sample.json` mirrors the exact
shape produced by `visualq/scripts/export-diff-bundles.ts` (anonymized ids,
nulls for unavailable blocks, backfill-approve labels). If schema evolution
breaks validation or featurization of this shape, the private exporter is
broken too — fail here, before an export run does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from visualq_vrai.features.extract import featurize_bundle
from visualq_vrai.schema.bundle import DiffBundle
from visualq_vrai.schema.feature_spec import FEATURE_COLUMNS, LABEL_COLUMN

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "export-samples" / "exported-sample.json"


def test_exporter_output_validates_and_featurizes():
    bundle = DiffBundle.model_validate(json.loads(SAMPLE.read_text(encoding="utf-8")))
    row = featurize_bundle(bundle)

    # Every schema column exists.
    for col in FEATURE_COLUMNS:
        assert col in row, f"missing feature column {col}"

    # Present evidence is featurized…
    assert not math.isnan(row["mismatch_pct_log"])
    assert not math.isnan(row["ci_present"]) and row["ci_present"] == 1.0
    assert row[LABEL_COLUMN] == "intentional_redesign"

    # …and absent blocks stay NaN (no connectors, no DOM map, no image, no history).
    assert math.isnan(row["commit_changed_since_baseline"])
    assert math.isnan(row["jira_key_present"])
    assert math.isnan(row["figma_alignment_delta"])
    assert math.isnan(row["interactive_elem_changed"])
    assert math.isnan(row["region_count_log"])
    assert math.isnan(row["mismatch_zscore"])
