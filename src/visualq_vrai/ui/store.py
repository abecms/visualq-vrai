"""Bundle store backing the review UI.

The bundle JSON files on disk are the single source of truth: verdicts are
written back into them atomically. TabICL triage results are cached in memory
and invalidated on every new verdict (the ICL context changed) — the UI then
shows heuristic triage again, explicitly badged as such. Never a silent
fallback: every summary carries its ``source``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from visualq_vrai.features.extract import bundles_to_dataframe
from visualq_vrai.heuristic.triage import triage_heuristic
from visualq_vrai.schema.bundle import DiffBundle, TriageClass
from visualq_vrai.schema.feature_spec import (
    LABEL_COLUMN,
    MIN_CLASS_EXAMPLES,
    MIN_CONTEXT_ROWS,
)

VALID_VERDICTS = {c.value for c in TriageClass.ordered()}


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class BundleStore:
    """Loads DiffBundles from a directory, serves triage, persists verdicts."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._bundles: dict[str, DiffBundle] = {}
        self._paths: dict[str, Path] = {}
        self._df: pd.DataFrame | None = None
        self._tabicl_cache: dict[str, dict[str, Any]] = {}
        self._tabicl_meta: dict[str, Any] = {"state": "not_run"}
        self.load()

    # -- loading -----------------------------------------------------------

    def load(self) -> None:
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Bundle directory not found: {self.directory}")
        bundles: dict[str, DiffBundle] = {}
        paths: dict[str, Path] = {}
        for path in sorted(self.directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            bundle = DiffBundle.model_validate(data)
            bundles[bundle.sampleId] = bundle
            paths[bundle.sampleId] = path
        self._bundles = bundles
        self._paths = paths
        self._df = None
        self._invalidate_tabicl("bundles_reloaded")

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._df = bundles_to_dataframe(list(self._bundles.values()))
        return self._df

    # -- triage ------------------------------------------------------------

    def _heuristic_row(self, row: dict[str, Any]) -> dict[str, Any]:
        h = triage_heuristic(row)
        probs = {c.value: 0.0 for c in TriageClass.ordered()}
        probs[h.class_name] = h.confidence
        return {
            "class": h.class_name,
            "probabilities": probs,
            "topFeatures": [{"feature": r, "shap": 0.0} for r in (h.reasons or [])],
            "conformal": {"autoTriage": False, "riskLevel": None, "deferReason": None},
            "source": "heuristic",
        }

    def triage_for(self, sample_id: str) -> dict[str, Any]:
        """TabICL result when cached, else heuristic — source always explicit."""
        cached = self._tabicl_cache.get(sample_id)
        if cached is not None:
            return cached
        df = self.dataframe
        row = df[df["sampleId"] == sample_id]
        if row.empty:
            raise KeyError(sample_id)
        return self._heuristic_row(row.iloc[0].to_dict())

    def refresh_tabicl(self) -> dict[str, Any]:
        """Run TabICL over all unlabeled bundles; cache per-sample results."""
        df = self.dataframe
        if LABEL_COLUMN not in df.columns:
            df[LABEL_COLUMN] = float("nan")
        context = df.dropna(subset=[LABEL_COLUMN])
        query = df[df[LABEL_COLUMN].isna()]

        status = self.label_status()
        if status["insufficientData"]:
            self._tabicl_meta = {
                "state": "insufficient",
                "reason": status["insufficientReason"],
                "computedAt": datetime.now(timezone.utc).isoformat(),
            }
            self._tabicl_cache = {}
            return self._tabicl_meta

        if query.empty:
            self._tabicl_meta = {
                "state": "no_query",
                "computedAt": datetime.now(timezone.utc).isoformat(),
            }
            return self._tabicl_meta

        from visualq_vrai.service.predict import predict_tabicl

        results = predict_tabicl(context, query)
        cache: dict[str, dict[str, Any]] = {}
        for sample_id, result in zip(query["sampleId"].tolist(), results):
            cache[sample_id] = {
                "class": result.class_name,
                "probabilities": {
                    k: _nan_to_none(v) for k, v in result.probabilities.items()
                },
                "topFeatures": result.top_features,
                "conformal": {
                    k: _nan_to_none(v) for k, v in result.conformal.items()
                },
                "source": result.source,
            }
        self._tabicl_cache = cache
        self._tabicl_meta = {
            "state": "ready",
            "queryCount": len(cache),
            "contextCount": int(len(context)),
            "computedAt": datetime.now(timezone.utc).isoformat(),
        }
        return self._tabicl_meta

    def _invalidate_tabicl(self, reason: str) -> None:
        self._tabicl_cache = {}
        self._tabicl_meta = {"state": "stale", "reason": reason}

    @property
    def tabicl_meta(self) -> dict[str, Any]:
        return self._tabicl_meta

    # -- label status ------------------------------------------------------

    def label_status(self) -> dict[str, Any]:
        counts = {c.value: 0 for c in TriageClass.ordered()}
        labeled = 0
        for bundle in self._bundles.values():
            if bundle.label and bundle.label.verdict:
                labeled += 1
                counts[bundle.label.verdict] += 1
        insufficient_reason: str | None = None
        if labeled < MIN_CONTEXT_ROWS:
            insufficient_reason = f"context_rows_below_{MIN_CONTEXT_ROWS}"
        else:
            for cls, count in counts.items():
                if count < MIN_CLASS_EXAMPLES:
                    insufficient_reason = f"class_{cls}_below_{MIN_CLASS_EXAMPLES}"
                    break
        return {
            "total": len(self._bundles),
            "labeled": labeled,
            "perClass": counts,
            "minContextRows": MIN_CONTEXT_ROWS,
            "minClassExamples": MIN_CLASS_EXAMPLES,
            "remainingToMinContext": max(MIN_CONTEXT_ROWS - labeled, 0),
            "insufficientData": insufficient_reason is not None,
            "insufficientReason": insufficient_reason,
        }

    # -- accessors ---------------------------------------------------------

    def get(self, sample_id: str) -> DiffBundle:
        bundle = self._bundles.get(sample_id)
        if bundle is None:
            raise KeyError(sample_id)
        return bundle

    def image_path(self, sample_id: str, kind: str) -> Path:
        bundle = self.get(sample_id)
        attr = {
            "baseline": bundle.images.baselinePng,
            "test": bundle.images.testPng,
            "diff": bundle.images.diffPng,
        }.get(kind)
        if attr is None:
            raise FileNotFoundError(f"No {kind} image for {sample_id}")
        path = Path(attr)
        if not path.is_absolute():
            path = _resolve_relative(path)
        if not path.is_file():
            raise FileNotFoundError(f"{kind} image missing on disk: {path}")
        return path

    def summaries(self) -> list[dict[str, Any]]:
        """One summary per bundle, sorted by regression probability descending."""
        out: list[dict[str, Any]] = []
        for sample_id, bundle in self._bundles.items():
            triage = self.triage_for(sample_id)
            reg_prob = triage["probabilities"].get(
                TriageClass.REGRESSION.value
            )
            out.append(
                {
                    "sampleId": sample_id,
                    "scenario": bundle.diff.scenario,
                    "viewport": bundle.diff.viewport,
                    "browser": bundle.diff.browser,
                    "misMatchPercentage": bundle.diff.misMatchPercentage,
                    "status": bundle.diff.status,
                    "verdict": bundle.label.verdict if bundle.label else None,
                    "triageClass": triage["class"],
                    "regressionProbability": _nan_to_none(reg_prob) or 0.0,
                    "source": triage["source"],
                    "hasImages": bundle.images.diffPng is not None,
                }
            )
        out.sort(key=lambda s: s["regressionProbability"], reverse=True)
        return out

    def detail(self, sample_id: str) -> dict[str, Any]:
        bundle = self.get(sample_id)
        triage = self.triage_for(sample_id)
        images = {}
        for kind in ("baseline", "test", "diff"):
            try:
                self.image_path(sample_id, kind)
                images[kind] = f"/ui/images/{sample_id}/{kind}"
            except (FileNotFoundError, KeyError):
                images[kind] = None
        intent = (
            bundle.intentSignals.model_dump(exclude_none=True)
            if bundle.intentSignals
            else None
        )
        return {
            "sampleId": sample_id,
            "scenario": bundle.diff.scenario,
            "viewport": bundle.diff.viewport,
            "browser": bundle.diff.browser,
            "status": bundle.diff.status,
            "misMatchPercentage": bundle.diff.misMatchPercentage,
            "diffPixels": bundle.diff.diffPixels,
            "dimensions": bundle.diff.dimensions,
            "hasShift": bundle.diff.hasShift,
            "shiftRegions": [
                r.model_dump() for r in (bundle.diff.shiftRegions or [])
            ],
            "runId": bundle.runContext.runId,
            "createdAt": bundle.runContext.createdAt.isoformat(),
            "ci": bundle.runContext.ci.model_dump() if bundle.runContext.ci else None,
            "intentSignals": intent,
            "historyCount": len(bundle.history),
            "verdict": bundle.label.verdict if bundle.label else None,
            "verdictSource": bundle.label.verdictSource if bundle.label else None,
            "verdictAt": (
                bundle.label.verdictAt.isoformat()
                if bundle.label and bundle.label.verdictAt
                else None
            ),
            "triage": triage,
            "images": images,
        }

    # -- verdicts ----------------------------------------------------------

    def set_verdict(self, sample_id: str, verdict: str) -> dict[str, Any]:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"Unknown verdict: {verdict}")
        self.get(sample_id)  # raises KeyError for unknown samples
        path = self._paths[sample_id]

        data = json.loads(path.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc)
        data["label"] = {
            "verdict": verdict,
            "verdictSource": "viewer",
            "verdictAt": now.isoformat(),
        }
        _atomic_write_json(path, data)

        updated = DiffBundle.model_validate(data)
        self._bundles[sample_id] = updated
        if self._df is not None:
            self._df.loc[self._df["sampleId"] == sample_id, LABEL_COLUMN] = verdict
        self._invalidate_tabicl("new_verdict")
        return {
            "sampleId": sample_id,
            "verdict": verdict,
            "verdictSource": "viewer",
            "verdictAt": now.isoformat(),
        }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _resolve_relative(path: Path) -> Path:
    """Resolve fixture-style relative paths from the repo root."""
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / path
    return candidate if candidate.is_file() else path
