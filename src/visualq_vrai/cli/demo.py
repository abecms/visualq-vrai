"""Hackathon demo CLI — replay fixtures with heuristic (+ TabICL when context sufficient)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from visualq_vrai.features.extract import bundles_to_dataframe, featurize_bundle
from visualq_vrai.heuristic.triage import triage_heuristic
from visualq_vrai.schema.bundle import DiffBundle
from visualq_vrai.schema.feature_spec import LABEL_COLUMN, MIN_CONTEXT_ROWS
from visualq_vrai.service.predict import predict_tabicl


def load_bundles(directory: Path) -> list[DiffBundle]:
    bundles: list[DiffBundle] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        bundles.append(DiffBundle.model_validate(data))
    return bundles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VRT diff triage demo")
    parser.add_argument("fixtures_dir", type=Path, help="Directory of DiffBundle JSON files")
    parser.add_argument("--query", type=str, default=None, help="sampleId to predict (default: unlabeled or last)")
    args = parser.parse_args(argv)

    bundles = load_bundles(args.fixtures_dir)
    if not bundles:
        print(f"No bundles found in {args.fixtures_dir}", file=sys.stderr)
        return 1

    df = bundles_to_dataframe(bundles)
    labeled = df.dropna(subset=[LABEL_COLUMN])
    query_bundle = None
    if args.query:
        query_bundle = next((b for b in bundles if b.sampleId == args.query), None)
        if query_bundle is None:
            print(f"sampleId not found: {args.query}", file=sys.stderr)
            return 1
    else:
        unlabeled = [b for b in bundles if not b.label or not b.label.verdict]
        query_bundle = unlabeled[-1] if unlabeled else bundles[-1]

    query_row = featurize_bundle(query_bundle)
    print(f"=== Query: {query_bundle.sampleId} ===")
    print(f"Scenario: {query_bundle.diff.scenario} | {query_bundle.diff.viewport} | {query_bundle.diff.browser}")
    print(f"Mismatch: {query_bundle.diff.misMatchPercentage:.2f}%")

    h = triage_heuristic(query_row)
    print(f"\n[Heuristic] {h.class_name} (confidence={h.confidence:.2f})")
    if h.reasons:
        print("  reasons:", ", ".join(h.reasons))

    if len(labeled) >= MIN_CONTEXT_ROWS:
        try:
            context_df = labeled.copy()
            query_df = bundles_to_dataframe([query_bundle])
            results = predict_tabicl(context_df, query_df)
            r = results[0]
            print(f"\n[TabICL] {r.class_name} (source={r.source})")
            print("  probabilities:", json.dumps(r.probabilities, indent=2))
            if r.top_features:
                print("  top SHAP features:")
                for feat in r.top_features[:5]:
                    print(f"    - {feat['feature']}: {feat['shap']:.4f}")
            print("  conformal:", r.conformal)
        except ImportError as exc:
            print(f"\n[TabICL] skipped — {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"\n[TabICL] error — {exc}", file=sys.stderr)
    else:
        print(f"\n[TabICL] insufficient labeled context ({len(labeled)} < {MIN_CONTEXT_ROWS})")

    if query_bundle.label and query_bundle.label.verdict:
        print(f"\n[Ground truth] {query_bundle.label.verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
