"""Rule-based baseline triage (Phase 0) — explicit product floor, never a silent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visualq_vrai.schema.bundle import TriageClass


@dataclass
class HeuristicResult:
    class_name: str
    confidence: float
    source: str = "heuristic"
    reasons: list[str] | None = None


def _get(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def triage_heuristic(row: dict[str, Any]) -> HeuristicResult:
    """Deterministic 3-class triage from ~8 high-signal features."""

    reasons: list[str] = []
    zscore = _get(row, "mismatch_zscore")
    isolated = _get(row, "isolated_pixel_ratio")
    only_browser = _get(row, "only_this_browser")
    run_fail_frac = _get(row, "run_fail_fraction")
    interactive = _get(row, "interactive_elem_changed")
    recurrence = _get(row, "spatial_recurrence_iou")
    commit_changed = _get(row, "commit_changed_since_baseline")
    figma_delta = _get(row, "figma_alignment_delta")

    # Platform constraint signatures
    platform_score = 0
    if only_browser == 1.0:
        platform_score += 2
        reasons.append("mono_browser_failure")
    if isolated > 0.15:
        platform_score += 2
        reasons.append("high_isolated_pixel_ratio")
    if commit_changed == 0.0:
        platform_score += 1
        reasons.append("no_code_change_since_baseline")
    if zscore < 1.0 and isolated > 0.05:
        platform_score += 1
        reasons.append("low_zscore_with_noise")

    if platform_score >= 3:
        confidence = min(0.55 + 0.08 * platform_score, 0.92)
        return HeuristicResult(
            class_name=TriageClass.PLATFORM.value,
            confidence=confidence,
            reasons=reasons,
        )

    # Intentional redesign signatures
    redesign_score = 0
    if run_fail_frac > 0.5:
        redesign_score += 2
        reasons.append("broad_run_failure")
    if figma_delta > 0.05:
        redesign_score += 2
        reasons.append("moving_toward_figma")
    if recurrence > 0.6:
        redesign_score += 1
        reasons.append("spatial_recurrence_with_prior_approve")

    if redesign_score >= 3:
        confidence = min(0.5 + 0.1 * redesign_score, 0.9)
        return HeuristicResult(
            class_name=TriageClass.INTENTIONAL.value,
            confidence=confidence,
            reasons=reasons,
        )

    # Regression signatures
    regression_score = 0
    if interactive == 1.0:
        regression_score += 2
        reasons.append("interactive_element_changed")
    if zscore > 2.0:
        regression_score += 2
        reasons.append("high_mismatch_zscore")
    if figma_delta < -0.05:
        regression_score += 2
        reasons.append("moving_away_from_figma")

    if regression_score >= 2:
        confidence = min(0.5 + 0.1 * regression_score, 0.88)
        return HeuristicResult(
            class_name=TriageClass.REGRESSION.value,
            confidence=confidence,
            reasons=reasons,
        )

    # Default: low-confidence regression (investigate)
    return HeuristicResult(
        class_name=TriageClass.REGRESSION.value,
        confidence=0.45,
        reasons=reasons or ["insufficient_heuristic_signal"],
    )
