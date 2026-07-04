"""TabICL prediction with guardrails."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vrt_triage.heuristic.triage import triage_heuristic
from vrt_triage.schema.bundle import TriageClass
from vrt_triage.schema.feature_spec import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    MIN_CLASS_EXAMPLES,
    MIN_CONTEXT_ROWS,
    SCHEMA_VERSION,
)
from vrt_triage.service.conformal import ConformalSelector


@dataclass
class PredictResult:
    class_name: str
    probabilities: dict[str, float]
    top_features: list[dict[str, float]]
    conformal: dict[str, Any]
    source: str
    insufficient_data: bool = False
    insufficient_reason: str | None = None


def _decode_parquet(b64: str) -> pd.DataFrame:
    raw = base64.b64decode(b64)
    return pd.read_parquet(io.BytesIO(raw))


def encode_parquet(df: pd.DataFrame) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _class_counts(labels: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cls in TriageClass.ordered():
        counts[cls.value] = int((labels == cls.value).sum())
    return counts


def _check_context(context_df: pd.DataFrame) -> str | None:
    if LABEL_COLUMN not in context_df.columns:
        return "missing_label_column"
    labeled = context_df.dropna(subset=[LABEL_COLUMN])
    if len(labeled) < MIN_CONTEXT_ROWS:
        return f"context_rows_below_{MIN_CONTEXT_ROWS}"
    counts = _class_counts(labeled[LABEL_COLUMN])
    for cls, count in counts.items():
        if count < MIN_CLASS_EXAMPLES:
            return f"class_{cls}_below_{MIN_CLASS_EXAMPLES}"
    return None


def _top_shap_features(shap_values: np.ndarray, feature_names: list[str], top_k: int = 5) -> list[dict[str, float]]:
    if shap_values.ndim == 1:
        values = shap_values
    else:
        values = shap_values[0]
    order = np.argsort(np.abs(values))[::-1][:top_k]
    return [
        {"feature": feature_names[i], "shap": float(values[i])}
        for i in order
        if not np.isnan(values[i])
    ]


def predict_tabicl(
    context_df: pd.DataFrame,
    query_df: pd.DataFrame,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> list[PredictResult]:
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schemaVersion mismatch: expected {SCHEMA_VERSION}, got {schema_version}")

    insufficiency = _check_context(context_df)
    query_rows = query_df.to_dict(orient="records")
    if insufficiency:
        return [
            PredictResult(
                class_name=triage_heuristic(row).class_name,
                probabilities={c.value: float("nan") for c in TriageClass.ordered()},
                top_features=[],
                conformal={"autoTriage": False, "riskLevel": float("nan")},
                source="heuristic",
                insufficient_data=True,
                insufficient_reason=insufficiency,
            )
            for row in query_rows
        ]

    from tabicl import TabICLClassifier

    labeled = context_df.dropna(subset=[LABEL_COLUMN]).copy()
    x_context = labeled[FEATURE_COLUMNS]
    y_context = labeled[LABEL_COLUMN].astype(str)
    x_query = query_df[FEATURE_COLUMNS]

    model = TabICLClassifier(kv_cache=True)
    model.fit(x_context, y_context)
    probas = model.predict_proba(x_query)
    classes = list(model.classes_)

    shap_values = None
    if hasattr(model, "explain"):
        shap_values = model.explain(x_query)
    elif hasattr(model, "shap"):
        shap_values = model.shap(x_query)

    # Calibration split: last 20% of labeled context by createdAt if present
    calib_df = labeled
    if "createdAt" in labeled.columns:
        calib_df = labeled.sort_values("createdAt")
    split = max(int(len(calib_df) * 0.8), MIN_CONTEXT_ROWS)
    calib_df = calib_df.iloc[split:]
    selector = ConformalSelector(nominal_risk=0.02)
    if len(calib_df) >= 30:
        calib_x = calib_df[FEATURE_COLUMNS]
        calib_y = calib_df[LABEL_COLUMN].astype(str)
        calib_proba = model.predict_proba(calib_x)
        y_idx = np.array([TriageClass.index(v) for v in calib_y])
        selector.fit(calib_proba, y_idx)
    else:
        selector.threshold_ = 0.5

    class_order = [c.value for c in TriageClass.ordered()]
    idx_map = {cls: i for i, cls in enumerate(classes)}

    results: list[PredictResult] = []
    decisions = selector.decide(probas)

    for i, row in enumerate(query_rows):
        prob_dict = {
            cls: float(probas[i, idx_map[cls]]) if cls in idx_map else 0.0
            for cls in class_order
        }
        pred_idx = int(np.argmax(probas[i]))
        pred_class = classes[pred_idx]

        top_features: list[dict[str, float]] = []
        if shap_values is not None:
            if isinstance(shap_values, list):
                top_features = _top_shap_features(np.array(shap_values[i]), FEATURE_COLUMNS)
            else:
                sv = np.array(shap_values)
                if sv.ndim == 3:
                    class_sv = sv[i, :, pred_idx]
                    top_features = _top_shap_features(class_sv, FEATURE_COLUMNS)
                elif sv.ndim == 2:
                    top_features = _top_shap_features(sv[i], FEATURE_COLUMNS)

        decision = decisions[i]
        results.append(
            PredictResult(
                class_name=pred_class,
                probabilities=prob_dict,
                top_features=top_features,
                conformal={
                    "autoTriage": decision.auto_triage,
                    "riskLevel": decision.risk_level,
                    "deferReason": decision.defer_reason,
                },
                source="tabicl",
            )
        )

    return results


def predict_heuristic(query_df: pd.DataFrame) -> list[PredictResult]:
    results: list[PredictResult] = []
    for row in query_df.to_dict(orient="records"):
        h = triage_heuristic(row)
        probs = {c.value: 0.0 for c in TriageClass.ordered()}
        probs[h.class_name] = h.confidence
        results.append(
            PredictResult(
                class_name=h.class_name,
                probabilities=probs,
                top_features=[{"feature": r, "shap": 0.0} for r in (h.reasons or [])],
                conformal={"autoTriage": False, "riskLevel": float("nan")},
                source="heuristic",
            )
        )
    return results
