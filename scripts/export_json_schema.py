"""Export JSON Schema for DiffBundle v1."""

import json
from pathlib import Path

from visualq_vrai.schema.bundle import DiffBundle

if __name__ == "__main__":
    schema = DiffBundle.model_json_schema()
    out = Path(__file__).resolve().parents[1] / "schema" / "diff-bundle.v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
