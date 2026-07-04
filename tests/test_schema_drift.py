"""Schema drift guard.

Intent: `schema/diff-bundle.v1.json` is the published contract that the
VisualQ exporter and bundle builder validate against. If the Pydantic model
changes without re-exporting (or without bumping bundleVersion), the two
sides silently diverge — this test fails loudly instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from visualq_vrai.schema.bundle import DiffBundle

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "diff-bundle.v1.json"


def test_exported_json_schema_matches_model():
    exported = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    current = DiffBundle.model_json_schema()
    assert exported == current, (
        "schema/diff-bundle.v1.json is stale — run "
        "`python scripts/export_json_schema.py` and review the diff for "
        "backward compatibility (new fields must be optional, or bump bundleVersion)"
    )
