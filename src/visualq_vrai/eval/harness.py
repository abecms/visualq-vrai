"""Evaluation harness — temporal + group splits, probabilistic metrics, ablations.

Every predictor is evaluated through the same interface (probabilities over
the three canonical classes) so heuristic, TabICL and LightGBM are directly
comparable, with the conformal selector applied identically on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

from visualq_vrai.heuristic.triage import triage_heuristic
from visualq_vrai.schema.bundle import TriageClass
from visualq_vrai.schema.feature_spec import FEATURE_COLUMNS, LABEL_COLUMN

CLASS_ORDER = [c.value for c in TriageClass.ordered()]
REG = TriageClass.REGRESSION.value
REG_IDX = TriageClass.index(REG)

# Feature groups joined from intent connectors — ablated as blocks.
INTENT_GROUPS: dict[str, list[str]] = {
    "github": [
        "commit_changed_since_baseline",
        "pr_files_changed_log",
        "ui_files_changed",
        "css_lines_changed_log",
        "pr_title_type_ord",
        "pr_has_design_label",
        "changed_files_match_page",
    ],
    "jira": [
        "jira_key_present",
        "jira_issue_is_story",
        "jira_issue_is_bug",
        "jira_has_design_label",
    ],
    "figma": [
        "figma_design_changed_recently",
        "days_since_figma_version_log",
        "figma_alignment_baseline",
        "figma_alignment_delta",
    ],
}

# predictor(train_df, test_df) -> probabilities of shape (len(test), 3)
# in canonical class order.
ProbaPredictor = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


@dataclass
class EvalMetrics:
    macro_f1: float
    regression_auc: float
    per_class_f1: dict[str, float]
    brier_regression: float
    precision_at_10: float
    auto_triage_fraction: float
    auto_triage_miss_rate: float
    n_train: int
    n_test: int
    notes: list[str] = field(default_factory=list)


def _predictions_from_probas(probas: np.ndarray) -> list[str]:
    return [CLASS_ORDER[int(np.argmax(p))] for p in probas]


def heuristic_proba_predictor(_train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    probas = np.zeros((len(test), 3))
    for i, row in enumerate(test.to_dict(orient="records")):
        h = triage_heuristic(row)
        residual = (1.0 - h.confidence) / 2.0
        probas[i, :] = residual
        probas[i, TriageClass.index(h.class_name)] = h.confidence
    return probas


def tabicl_proba_predictor(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from tabicl import TabICLClassifier

    from visualq_vrai.service.predict import _active_columns

    active = _active_columns(train, test)
    model = TabICLClassifier(kv_cache=True)
    model.fit(train[active], train[LABEL_COLUMN].astype(str))
    raw = model.predict_proba(test[active])
    idx_map = {cls: i for i, cls in enumerate(model.classes_)}
    return np.column_stack([
        raw[:, idx_map[cls]] if cls in idx_map else np.zeros(len(raw))
        for cls in CLASS_ORDER
    ])


def lightgbm_proba_predictor(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Head-to-head baseline. Requires the optional [bench] extra."""
    from lightgbm import LGBMClassifier

    x_train = train[FEATURE_COLUMNS]
    y_train = train[LABEL_COLUMN].astype(str)
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=0,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    raw = model.predict_proba(test[FEATURE_COLUMNS])
    idx_map = {cls: i for i, cls in enumerate(model.classes_)}
    return np.column_stack([
        raw[:, idx_map[cls]] if cls in idx_map else np.zeros(len(raw))
        for cls in CLASS_ORDER
    ])


def temporal_split(df: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df.dropna(subset=[LABEL_COLUMN]).copy()
    if "createdAt" in labeled.columns:
        labeled = labeled.sort_values("createdAt")
    split = max(int(len(labeled) * train_fraction), 1)
    train, test = labeled.iloc[:split], labeled.iloc[split:]
    if len(test) == 0:
        test = train.iloc[-max(len(train) // 5, 1):]
        train = train.iloc[: len(train) - len(test)]
    return train, test


def group_split(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire groups (e.g. scenarios) — measures generalization to
    pages the model has never seen, which a temporal split cannot."""
    labeled = df.dropna(subset=[LABEL_COLUMN]).copy()
    groups = sorted(labeled[group_col].dropna().unique())
    if len(groups) < 2:
        raise ValueError(f"group_split needs >=2 values in {group_col!r}, got {groups}")
    n_holdout = max(len(groups) // 5, 1)
    holdout = set(groups[-n_holdout:])
    train = labeled[~labeled[group_col].isin(holdout)]
    test = labeled[labeled[group_col].isin(holdout)]
    return train, test


def _conformal_stats(
    train: pd.DataFrame,
    test_probas: np.ndarray,
    test_labels: pd.Series,
    predictor: ProbaPredictor,
    nominal_risk: float = 0.02,
) -> tuple[float, float, list[str]]:
    """Fit the selector on a held-out slice of train, measure auto-triage
    fraction and empirical missed-regression rate on test."""
    from visualq_vrai.service.conformal import ConformalSelector
    from visualq_vrai.service.predict import MIN_CALIBRATION_ROWS

    notes: list[str] = []
    holdout = max(int(len(train) * 0.15), MIN_CALIBRATION_ROWS)
    if len(train) - holdout < 30:
        return 0.0, float("nan"), ["calibration_too_small"]
    fit_df, calib_df = train.iloc[:-holdout], train.iloc[-holdout:]
    calib_probas = predictor(fit_df, calib_df)
    y_calib = np.array([TriageClass.index(v) for v in calib_df[LABEL_COLUMN].astype(str)])
    selector = ConformalSelector(nominal_risk=nominal_risk)
    selector.fit(calib_probas, y_calib)
    decisions = selector.decide(test_probas)
    auto_idx = [i for i, d in enumerate(decisions) if d.auto_triage]
    fraction = len(auto_idx) / max(len(decisions), 1)
    if auto_idx:
        y_test = test_labels.to_numpy()
        miss = sum(1 for i in auto_idx if y_test[i] == REG) / len(auto_idx)
    else:
        miss = float("nan")
        notes.append("no_auto_triage")
    return fraction, miss, notes


def evaluate_probabilistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictor: ProbaPredictor,
    *,
    with_conformal: bool = True,
) -> EvalMetrics:
    probas = predictor(train, test)
    preds = _predictions_from_probas(probas)
    y_true = test[LABEL_COLUMN].astype(str)

    macro = float(f1_score(y_true, preds, average="macro", zero_division=0))
    per_class = {
        cls: float(f1_score(y_true, preds, labels=[cls], average="macro", zero_division=0))
        for cls in CLASS_ORDER
    }

    reg_binary = (y_true == REG).astype(int)
    reg_scores = probas[:, REG_IDX]
    auc = (
        float(roc_auc_score(reg_binary, reg_scores))
        if reg_binary.nunique() > 1
        else float("nan")
    )
    brier = (
        float(brier_score_loss(reg_binary, np.clip(reg_scores, 0, 1)))
        if len(reg_binary)
        else float("nan")
    )

    k = min(10, len(test))
    if k and reg_binary.sum() > 0:
        top_k = np.argsort(reg_scores)[::-1][:k]
        precision_at_10 = float(reg_binary.to_numpy()[top_k].mean())
    else:
        precision_at_10 = float("nan")

    fraction, miss, notes = (0.0, float("nan"), [])
    if with_conformal:
        fraction, miss, notes = _conformal_stats(train, probas, y_true, predictor)

    return EvalMetrics(
        macro_f1=macro,
        regression_auc=auc,
        per_class_f1=per_class,
        brier_regression=brier,
        precision_at_10=precision_at_10,
        auto_triage_fraction=fraction,
        auto_triage_miss_rate=miss,
        n_train=len(train),
        n_test=len(test),
        notes=notes,
    )


def ablate_intent_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Return a copy with one connector's columns NaN-ised — measures what
    that connector contributes, without touching anything else."""
    columns = INTENT_GROUPS[group]
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = float("nan")
    return out


# -- backward-compatible convenience wrappers --------------------------------


def evaluate_heuristic(df: pd.DataFrame) -> EvalMetrics:
    train, test = temporal_split(df)
    return evaluate_probabilistic(train, test, heuristic_proba_predictor, with_conformal=False)


def evaluate_group_holdout(df: pd.DataFrame, group_col: str = "projectId") -> EvalMetrics:
    labeled = df.dropna(subset=[LABEL_COLUMN])
    if len(labeled[group_col].dropna().unique()) < 2:
        return evaluate_heuristic(df)
    train, test = group_split(df, group_col)
    return evaluate_probabilistic(train, test, heuristic_proba_predictor, with_conformal=False)
