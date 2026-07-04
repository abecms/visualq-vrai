#!/usr/bin/env python3
"""Run evaluation harness on fixture bundles."""

from __future__ import annotations

import json
from pathlib import Path

from vrt_triage.eval.harness import evaluate_group_holdout, evaluate_heuristic
from vrt_triage.features.extract import bundles_to_dataframe
from vrt_triage.schema.bundle import DiffBundle


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    bundles_dir = root / "fixtures" / "bundles"
    bundles = [
        DiffBundle.model_validate(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(bundles_dir.glob("*.json"))
    ]
    df = bundles_to_dataframe(bundles)
    temporal = evaluate_heuristic(df)
    group = evaluate_group_holdout(df)
    print("Temporal split:", temporal)
    print("Group holdout:", group)


if __name__ == "__main__":
    main()
