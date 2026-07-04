"""Evaluation harness — temporal + group splits, heuristic floor benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from visualq_vrai.heuristic.triage import triage_heuristic
from visualq_vrai.schema.bundle import TriageClass
from visualq_vrai.schema.feature_spec import LABEL_COLUMN


@dataclass
class EvalMetrics:
    macro_f1: float
    regression_auc: float
    per_class_f1: dict[str, float]
    n_train: int
    n_test: int


def _binary_regression_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    binary = (y_true == TriageClass.REGRESSION.value).astype(int)
    if binary.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(binary, y_score))


def evaluate_predictor(
    df: pd.DataFrame,
    predictor: Callable[[pd.DataFrame, pd.DataFrame], list[str]],
    *,
    temporal: bool = True,
) -> EvalMetrics:
    labeled = df.dropna(subset=[LABEL_COLUMN]).copy()
    if "createdAt" in labeled.columns:
        labeled = labeled.sort_values("createdAt")
    split = max(int(len(labeled) * 0.8), 1)
    train = labeled.iloc[:split]
    test = labeled.iloc[split:]
    if len(test) == 0:
        test = train.iloc[-max(len(train) // 5, 1):]
        train = train.iloc[: len(train) - len(test)]

    preds = predictor(train, test)
    y_true = test[LABEL_COLUMN].astype(str)
    macro = float(f1_score(y_true, preds, average="macro", zero_division=0))
    per_class = {
        cls.value: float(f1_score(y_true, preds, labels=[cls.value], average="macro", zero_division=0))
        for cls in TriageClass.ordered()
    }
    reg_scores = np.array([
        1.0 if p == TriageClass.REGRESSION.value else 0.0 for p in preds
    ])
    auc = _binary_regression_auc(y_true, reg_scores)
    return EvalMetrics(
        macro_f1=macro,
        regression_auc=auc,
        per_class_f1=per_class,
        n_train=len(train),
        n_test=len(test),
    )


def evaluate_heuristic(df: pd.DataFrame) -> EvalMetrics:
    def predict(_train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
        return [triage_heuristic(row).class_name for row in test.to_dict(orient="records")]

    return evaluate_predictor(df, predict)


def evaluate_group_holdout(
    df: pd.DataFrame,
    group_col: str = "projectId",
) -> EvalMetrics:
    labeled = df.dropna(subset=[LABEL_COLUMN]).copy()
    groups = labeled[group_col].dropna().unique()
    if len(groups) < 2:
        return evaluate_heuristic(labeled)

    holdout = groups[-1]
    train = labeled[labeled[group_col] != holdout]
    test = labeled[labeled[group_col] == holdout]

    def predict(_train: pd.DataFrame, _test: pd.DataFrame) -> list[str]:
        return [triage_heuristic(row).class_name for row in test.to_dict(orient="records")]

    preds = predict(train, test)
    y_true = test[LABEL_COLUMN].astype(str)
    macro = float(f1_score(y_true, preds, average="macro", zero_division=0))
    per_class = {
        cls.value: float(f1_score(y_true, preds, labels=[cls.value], average="macro", zero_division=0))
        for cls in TriageClass.ordered()
    }
    reg_scores = np.array([
        1.0 if p == TriageClass.REGRESSION.value else 0.0 for p in preds
    ])
    auc = _binary_regression_auc(y_true, reg_scores)
    return EvalMetrics(
        macro_f1=macro,
        regression_auc=auc,
        per_class_f1=per_class,
        n_train=len(train),
        n_test=len(test),
    )
