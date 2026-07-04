"""TabICL prediction with guardrails."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from visualq_vrai.heuristic.triage import triage_heuristic
from visualq_vrai.schema.bundle import TriageClass
from visualq_vrai.schema.feature_spec import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    MIN_CLASS_EXAMPLES,
    MIN_CONTEXT_ROWS,
    SCHEMA_VERSION,
)
from visualq_vrai.service.conformal import ConformalDecision, ConformalSelector

# Minimum held-out rows required to calibrate the conformal selector. Below
# this, predictions are served but auto-triage is disabled for the whole batch
# (deferReason: calibration_too_small) — never a guessed threshold.
MIN_CALIBRATION_ROWS = 40
CALIBRATION_FRACTION = 0.15


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


def _active_columns(context_df: pd.DataFrame, query_df: pd.DataFrame) -> list[str]:
    """Columns with at least one observed value in context or query (TabICL drops all-NaN cols)."""
    active: list[str] = []
    for col in FEATURE_COLUMNS:
        ctx_obs = col in context_df.columns and context_df[col].notna().any()
        qry_obs = col in query_df.columns and query_df[col].notna().any()
        if ctx_obs or qry_obs:
            active.append(col)
    if not active:
        raise ValueError("No active feature columns — context and query are entirely NaN")
    return active


def _top_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    class_index: int,
    top_k: int = 5,
) -> list[dict[str, float]]:
    values = np.asarray(shap_values)
    if values.ndim == 3:
        row_values = values[0, :, class_index]
    elif values.ndim == 2:
        row_values = values[0] if values.shape[0] == 1 else values[:, class_index]
    else:
        row_values = values
    order = np.argsort(np.abs(row_values))[::-1][:top_k]
    return [
        {"feature": feature_names[i], "shap": float(row_values[i])}
        for i in order
        if i < len(feature_names) and not np.isnan(row_values[i])
    ]


def _compute_shap(model, x_query: pd.DataFrame, feature_names: list[str], class_indices: list[int]) -> list[list[dict[str, float]]]:
    from tabicl.shap import get_shap_values

    explanation = get_shap_values(model, x_query, attribute_names=feature_names)
    values = np.asarray(explanation.values)
    results: list[list[dict[str, float]]] = []
    for i, pred_idx in enumerate(class_indices):
        if values.ndim == 3:
            row_values = values[i, :, pred_idx]
        elif values.ndim == 2:
            row_values = values[i]
        else:
            row_values = values
        order = np.argsort(np.abs(row_values))[::-1][:5]
        results.append([
            {"feature": feature_names[j], "shap": float(row_values[j])}
            for j in order
            if j < len(feature_names) and not np.isnan(row_values[j])
        ])
    return results


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
                conformal={
                    "autoTriage": False,
                    "riskLevel": float("nan"),
                    "deferReason": "insufficient_data",
                },
                source="heuristic",
                insufficient_data=True,
                insufficient_reason=insufficiency,
            )
            for row in query_rows
        ]

    from tabicl import TabICLClassifier

    labeled = context_df.dropna(subset=[LABEL_COLUMN]).copy()
    if "createdAt" in labeled.columns:
        labeled = labeled.sort_values("createdAt")

    # Held-out calibration: the most recent verdicts never enter the ICL fit.
    # Only carved out when the remaining fit set still satisfies MIN_CONTEXT_ROWS;
    # otherwise predictions are served with auto-triage disabled for the batch.
    holdout_size = max(int(len(labeled) * CALIBRATION_FRACTION), MIN_CALIBRATION_ROWS)
    calibration_possible = len(labeled) - holdout_size >= MIN_CONTEXT_ROWS
    if calibration_possible:
        fit_df = labeled.iloc[:-holdout_size]
        calib_df = labeled.iloc[-holdout_size:]
    else:
        fit_df = labeled
        calib_df = labeled.iloc[0:0]

    active_cols = _active_columns(labeled, query_df)
    x_query = query_df[active_cols]

    model = TabICLClassifier(kv_cache=True)
    model.fit(fit_df[active_cols], fit_df[LABEL_COLUMN].astype(str))
    probas = model.predict_proba(x_query)
    classes = list(model.classes_)

    pred_indices = [int(np.argmax(probas[i])) for i in range(len(probas))]
    shap_by_row = _compute_shap(model, x_query, active_cols, pred_indices)

    class_order = [c.value for c in TriageClass.ordered()]
    idx_map = {cls: i for i, cls in enumerate(classes)}
    # Selector operates in canonical class order; remap model output.
    canonical_probas = np.column_stack([
        probas[:, idx_map[cls]] if cls in idx_map else np.zeros(len(probas))
        for cls in class_order
    ])

    if calibration_possible:
        selector = ConformalSelector(nominal_risk=0.02)
        calib_proba = model.predict_proba(calib_df[active_cols])
        canonical_calib = np.column_stack([
            calib_proba[:, idx_map[cls]] if cls in idx_map else np.zeros(len(calib_proba))
            for cls in class_order
        ])
        y_idx = np.array([
            TriageClass.index(v) for v in calib_df[LABEL_COLUMN].astype(str)
        ])
        selector.fit(canonical_calib, y_idx)
        decisions = selector.decide(canonical_probas)
    else:
        decisions = [
            ConformalDecision(
                auto_triage=False,
                risk_level=float(canonical_probas[i, TriageClass.index(TriageClass.REGRESSION.value)]),
                predicted_class=classes[int(np.argmax(probas[i]))],
                defer_reason="calibration_too_small",
            )
            for i in range(len(probas))
        ]

    results: list[PredictResult] = []

    for i, row in enumerate(query_rows):
        prob_dict = {
            cls: float(probas[i, idx_map[cls]]) if cls in idx_map else 0.0
            for cls in class_order
        }
        pred_idx = int(np.argmax(probas[i]))
        pred_class = classes[pred_idx]

        top_features = shap_by_row[i] if i < len(shap_by_row) else []

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
                conformal={
                    "autoTriage": False,
                    "riskLevel": float("nan"),
                    "deferReason": None,
                },
                source="heuristic",
            )
        )
    return results
