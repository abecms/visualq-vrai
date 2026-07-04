#!/usr/bin/env python3
"""Run the evaluation harness on fixture bundles.

Usage:
    python eval/run_eval.py [--bundles DIR] [--tabicl] [--lightgbm] [--ablation]

Heuristic metrics are always reported (fast). TabICL / LightGBM and the
connector ablation are opt-in because they are slow or need extras.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from visualq_vrai.eval.harness import (
    INTENT_GROUPS,
    ablate_intent_group,
    evaluate_probabilistic,
    group_split,
    heuristic_proba_predictor,
    lightgbm_proba_predictor,
    tabicl_proba_predictor,
    temporal_split,
)
from visualq_vrai.features.extract import bundles_to_dataframe
from visualq_vrai.schema.bundle import DiffBundle


def _load(bundles_dir: Path):
    bundles = [
        DiffBundle.model_validate(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(bundles_dir.glob("*.json"))
    ]
    return bundles_to_dataframe(bundles)


def _report(name: str, metrics) -> None:
    payload = asdict(metrics)
    print(f"\n=== {name} ===")
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, default=root / "fixtures" / "bundles")
    parser.add_argument("--tabicl", action="store_true", help="Also benchmark TabICL (slow)")
    parser.add_argument("--lightgbm", action="store_true", help="Also benchmark LightGBM ([bench] extra)")
    parser.add_argument("--ablation", action="store_true", help="Connector ablation (needs --tabicl or heuristic only)")
    args = parser.parse_args()

    df = _load(args.bundles)
    train_t, test_t = temporal_split(df)

    _report("Heuristic — temporal split", evaluate_probabilistic(
        train_t, test_t, heuristic_proba_predictor, with_conformal=False
    ))
    try:
        train_g, test_g = group_split(df, "scenario" if "scenario" in df.columns else "projectId")
        _report("Heuristic — group holdout", evaluate_probabilistic(
            train_g, test_g, heuristic_proba_predictor, with_conformal=False
        ))
    except ValueError as exc:
        print(f"\n[group holdout skipped: {exc}]")

    predictors = {}
    if args.tabicl:
        predictors["TabICL"] = tabicl_proba_predictor
    if args.lightgbm:
        predictors["LightGBM"] = lightgbm_proba_predictor

    for name, predictor in predictors.items():
        _report(f"{name} — temporal split (+conformal)", evaluate_probabilistic(
            train_t, test_t, predictor
        ))

    if args.ablation:
        predictor = predictors.get("TabICL", heuristic_proba_predictor)
        base = evaluate_probabilistic(train_t, test_t, predictor, with_conformal=False)
        print(f"\n=== Connector ablation (baseline AUC {base.regression_auc:.4f}) ===")
        for group in INTENT_GROUPS:
            ablated_train = ablate_intent_group(train_t, group)
            ablated_test = ablate_intent_group(test_t, group)
            m = evaluate_probabilistic(ablated_train, ablated_test, predictor, with_conformal=False)
            delta = base.regression_auc - m.regression_auc
            print(f"  without {group:<7} AUC {m.regression_auc:.4f} (contribution {delta:+.4f})")


if __name__ == "__main__":
    main()
