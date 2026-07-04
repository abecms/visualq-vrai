"""Split conformal risk control for 3-class auto-triage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visualq_vrai.schema.bundle import TriageClass


@dataclass
class ConformalDecision:
    auto_triage: bool
    risk_level: float
    predicted_class: str
    defer_reason: str | None = None


class ConformalSelector:
    """Defer predictions when regression risk exceeds nominal level on calibration set."""

    def __init__(self, nominal_risk: float = 0.02, regression_index: int | None = None):
        self.nominal_risk = nominal_risk
        self.regression_index = regression_index if regression_index is not None else TriageClass.index(
            TriageClass.REGRESSION.value
        )
        self.threshold_: float | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> None:
        """Calibrate the largest P(regression) cutoff whose accepted set keeps
        the true-regression rate at or below the nominal risk.

        Auto-triage means "trust the model without human review" — safe only
        for cases the model considers unlikely to be regressions. Cases are
        ranked by ascending P(regression); the cutoff is the largest prefix
        whose empirical regression rate stays within budget.
        """
        if len(probabilities) == 0:
            self.threshold_ = None
            return
        scores = probabilities[:, self.regression_index]
        is_regression = y_true == self.regression_index
        order = np.argsort(scores)
        scores_sorted = scores[order]
        labels_sorted = is_regression[order]
        n = len(scores_sorted)
        # Finite-sample conformal risk control: the "+1" upper-bounds the risk
        # of the next exchangeable case, keeping the guarantee on unseen data.
        # It also forces a minimum prefix size (~1/alpha clean cases) before
        # any auto-triage is allowed — tiny lucky prefixes cannot qualify.
        cumulative_risk = (np.cumsum(labels_sorted) + 1.0) / (
            np.arange(1, n + 1) + 1.0
        )
        eligible = np.where(cumulative_risk <= self.nominal_risk)[0]
        if len(eligible) == 0:
            self.threshold_ = -1.0
            return
        cutoff_idx = eligible[-1]
        self.threshold_ = float(scores_sorted[cutoff_idx])

    def decide(self, probabilities: np.ndarray) -> list[ConformalDecision]:
        if self.threshold_ is None:
            raise RuntimeError("ConformalSelector.fit must be called before decide")

        decisions: list[ConformalDecision] = []
        for probs in probabilities:
            pred_idx = int(np.argmax(probs))
            pred_class = TriageClass.ordered()[pred_idx].value
            score = float(probs[self.regression_index])
            if self.threshold_ < 0 or score > self.threshold_:
                decisions.append(
                    ConformalDecision(
                        auto_triage=False,
                        risk_level=float(probs[self.regression_index]),
                        predicted_class=pred_class,
                        defer_reason="conformal_defer",
                    )
                )
            else:
                decisions.append(
                    ConformalDecision(
                        auto_triage=True,
                        risk_level=float(probs[self.regression_index]),
                        predicted_class=pred_class,
                    )
                )
        return decisions
