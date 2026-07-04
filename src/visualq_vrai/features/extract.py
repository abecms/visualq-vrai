"""Featurize DiffBundle → tabular row (~86 columns)."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from visualq_vrai.features.spatial import extract_spatial_features
from visualq_vrai.schema.bundle import DiffBundle, ElementDiffResult
from visualq_vrai.schema.feature_spec import (
    AI_CATEGORIES,
    BROWSERS,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    PR_TITLE_TYPES,
    SCHEMA_VERSION,
    TRIGGERS,
    VIEWPORT_CLASSES,
)


def _log1p(value: float | None) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return math.log1p(max(value, 0.0))


def _ordinal(value: str | None, vocabulary: list[str]) -> float:
    if value is None:
        return float("nan")
    normalized = value.strip().lower()
    if normalized not in vocabulary:
        normalized = vocabulary[-1]
    return float(vocabulary.index(normalized))


def _bool_float(value: bool | None) -> float:
    if value is None:
        return float("nan")
    return 1.0 if value else 0.0


def _viewport_class(width: float) -> str:
    if width < 768:
        return "mobile"
    if width < 1024:
        return "tablet"
    if width < 1440:
        return "desktop"
    return "wide"


INTERACTIVE_TAGS = {"button", "a", "input", "select", "textarea"}


def _section_affected(elements: list[ElementDiffResult] | None, keywords: set[str]) -> float:
    if not elements:
        return float("nan")
    for elem in elements:
        if elem.status in {"unchanged", "ignored", "dynamic"}:
            continue
        haystack = " ".join(
            filter(
                None,
                [
                    elem.tag,
                    elem.semanticLabel or "",
                    elem.ariaLabel or "",
                    elem.selector,
                ],
            )
        ).lower()
        if any(k in haystack for k in keywords):
            return 1.0
    return 0.0


def _historical_stats(bundle: DiffBundle) -> dict[str, float]:
    history = sorted(bundle.history, key=lambda h: h.createdAt)
    if len(history) < 3:
        return {k: float("nan") for k in [
            "hist_fail_rate",
            "hist_approval_rate",
            "hist_regression_rate",
            "hist_mismatch_mean_log",
            "hist_mismatch_std_log",
            "mismatch_zscore",
            "runs_since_baseline_update",
            "days_since_last_fail_log",
            "spatial_recurrence_iou",
            "consecutive_fail_streak",
        ]}

    fails = [1.0 if h.status in {"fail", "failed"} else 0.0 for h in history]
    approvals = [1.0 if h.verdict == "intentional_redesign" else 0.0 for h in history]
    regressions = [1.0 if h.verdict == "accidental_regression" else 0.0 for h in history]
    mismatches = [h.misMatchPercentage for h in history]
    log_mismatches = [_log1p(m) for m in mismatches]
    mean_log = float(np.nanmean(log_mismatches))
    std_log = float(np.nanstd(log_mismatches)) if len(log_mismatches) > 1 else 0.0
    current_log = _log1p(bundle.diff.misMatchPercentage)
    zscore = (current_log - mean_log) / std_log if std_log > 1e-9 else 0.0

    streak = 0
    for h in reversed(history):
        if h.status in {"fail", "failed"}:
            streak += 1
        else:
            break

    last_fail: datetime | None = None
    for h in reversed(history):
        if h.status in {"fail", "failed"}:
            last_fail = h.createdAt
            break
    days_since_fail = float("nan")
    if last_fail is not None:
        delta = bundle.runContext.createdAt - last_fail
        days_since_fail = _log1p(delta.total_seconds() / 86400.0)

    runs_since_baseline = float("nan")
    for i, h in enumerate(reversed(history)):
        if h.verdict == "intentional_redesign":
            runs_since_baseline = float(i)
            break

    return {
        "hist_fail_rate": float(np.mean(fails)),
        "hist_approval_rate": float(np.mean(approvals)),
        "hist_regression_rate": float(np.mean(regressions)),
        "hist_mismatch_mean_log": mean_log,
        "hist_mismatch_std_log": std_log,
        "mismatch_zscore": zscore,
        "runs_since_baseline_update": runs_since_baseline,
        "days_since_last_fail_log": days_since_fail,
        "spatial_recurrence_iou": float("nan"),
        "consecutive_fail_streak": float(streak),
    }


def _cross_view_features(bundle: DiffBundle) -> dict[str, float]:
    siblings = bundle.runContext.siblingResults
    if not siblings:
        return {
            "viewport_fail_fraction": float("nan"),
            "browser_fail_fraction": float("nan"),
            "only_this_browser": float("nan"),
            "run_fail_fraction": float("nan"),
            "same_zone_cross_page_fraction": float("nan"),
        }

    same_scenario = [s for s in siblings if s.scenario == bundle.diff.scenario]
    viewport_fails = [1.0 if s.status == "fail" else 0.0 for s in same_scenario]
    browser_fails = [
        1.0 if s.status == "fail" else 0.0
        for s in same_scenario
        if s.viewport == bundle.diff.viewport
    ]
    all_fails = [1.0 if s.status == "fail" else 0.0 for s in siblings]
    failing_browsers = {
        s.browser for s in same_scenario if s.viewport == bundle.diff.viewport and s.status == "fail"
    }
    only_this_browser = (
        1.0
        if len(failing_browsers) == 1 and bundle.diff.browser in failing_browsers
        else 0.0
    )

    return {
        "viewport_fail_fraction": float(np.mean(viewport_fails)) if viewport_fails else float("nan"),
        "browser_fail_fraction": float(np.mean(browser_fails)) if browser_fails else float("nan"),
        "only_this_browser": only_this_browser,
        "run_fail_fraction": float(np.mean(all_fails)) if all_fails else float("nan"),
        "same_zone_cross_page_fraction": float("nan"),
    }


def _intent_github(bundle: DiffBundle) -> dict[str, float]:
    gh = bundle.intentSignals.github if bundle.intentSignals else None
    if gh is None:
        return {
            "commit_changed_since_baseline": float("nan"),
            "pr_files_changed_log": float("nan"),
            "ui_files_changed": float("nan"),
            "css_lines_changed_log": float("nan"),
            "pr_title_type_ord": float("nan"),
            "pr_has_design_label": float("nan"),
            "changed_files_match_page": float("nan"),
        }
    return {
        "commit_changed_since_baseline": _bool_float(gh.commitChangedSinceBaseline),
        "pr_files_changed_log": _log1p(gh.prFilesChanged),
        "ui_files_changed": _bool_float(gh.uiFilesChanged),
        "css_lines_changed_log": _log1p(gh.cssLinesChanged),
        "pr_title_type_ord": _ordinal(gh.prTitleType, PR_TITLE_TYPES),
        "pr_has_design_label": _bool_float(gh.prHasDesignLabel),
        "changed_files_match_page": _bool_float(gh.changedFilesMatchPage),
    }


def _intent_jira(bundle: DiffBundle) -> dict[str, float]:
    jira = bundle.intentSignals.jira if bundle.intentSignals else None
    if jira is None:
        return {
            "jira_key_present": float("nan"),
            "jira_issue_is_story": float("nan"),
            "jira_issue_is_bug": float("nan"),
            "jira_has_design_label": float("nan"),
        }
    return {
        "jira_key_present": _bool_float(jira.jiraKeyPresent),
        "jira_issue_is_story": _bool_float(jira.jiraIssueIsStory),
        "jira_issue_is_bug": _bool_float(jira.jiraIssueIsBug),
        "jira_has_design_label": _bool_float(jira.jiraHasDesignLabel),
    }


def _intent_figma(bundle: DiffBundle) -> dict[str, float]:
    figma = bundle.intentSignals.figma if bundle.intentSignals else None
    if figma is None:
        return {
            "figma_design_changed_recently": float("nan"),
            "days_since_figma_version_log": float("nan"),
            "figma_alignment_baseline": float("nan"),
            "figma_alignment_delta": float("nan"),
        }
    return {
        "figma_design_changed_recently": _bool_float(figma.figmaDesignChangedRecently),
        "days_since_figma_version_log": _log1p(figma.daysSinceFigmaVersion),
        "figma_alignment_baseline": (
            float(figma.figmaAlignmentBaseline)
            if figma.figmaAlignmentBaseline is not None
            else float("nan")
        ),
        "figma_alignment_delta": (
            float(figma.figmaAlignmentDelta)
            if figma.figmaAlignmentDelta is not None
            else float("nan")
        ),
    }


def _ai_features(bundle: DiffBundle) -> dict[str, float]:
    ai = bundle.diff.aiAnalysis
    if ai is None:
        return {
            "ai_severity_ord": float("nan"),
            "ai_category_ord": float("nan"),
            "ai_confidence": float("nan"),
            "ai_action_ord": float("nan"),
        }
    severity_map = {"cosmetic": 0.0, "minor": 1.0, "major": 2.0, "critical": 3.0}
    action_map = {"approve": 0.0, "investigate": 1.0, "reject": 2.0}
    return {
        "ai_severity_ord": severity_map.get(ai.severity, float("nan")),
        "ai_category_ord": _ordinal(ai.category, AI_CATEGORIES),
        "ai_confidence": float(ai.confidence),
        "ai_action_ord": action_map.get(ai.actionRecommended, float("nan")),
    }


def featurize_bundle(bundle: DiffBundle) -> dict[str, Any]:
    diff = bundle.diff
    width = float(diff.dimensions.get("width", 0))
    height = float(diff.dimensions.get("height", 0))
    effective_total = diff.effectiveTotalPixels or diff.totalPixels or 1.0
    baseline_height = diff.baselineHeight or height

    diff_png = bundle.images.diffPng
    if diff_png and not Path(diff_png).is_file():
        # Resolve relative paths from repo root (fixtures/…)
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / diff_png
        if candidate.is_file():
            diff_png = str(candidate)
    spatial = extract_spatial_features(diff_png, int(width), int(height))

    elements = diff.elementResults or []
    status_counts = {
        "visual_change": 0,
        "added": 0,
        "removed": 0,
        "moved": 0,
        "layout_change": 0,
        "dynamic": 0,
    }
    confidences: list[float] = []
    max_elem_pct = 0.0
    dynamic_diff_pixels = 0.0
    for elem in elements:
        if elem.status in status_counts:
            status_counts[elem.status] += 1
        if elem.diffPercentage is not None:
            max_elem_pct = max(max_elem_pct, elem.diffPercentage)
        if elem.matchConfidence is not None:
            confidences.append(elem.matchConfidence)
        if elem.status == "dynamic" and elem.diffPixelCount:
            dynamic_diff_pixels += elem.diffPixelCount

    interactive_changed = 0.0
    if elements:
        interactive_changed = 1.0 if any(
            e.tag.lower() in INTERACTIVE_TAGS and e.status not in {"unchanged", "ignored"}
            for e in elements
        ) else 0.0

    scenario_cfg = bundle.runContext.scenarioConfig or {}
    comparison_rules = scenario_cfg.get("comparisonRules") or []
    content_rules = scenario_cfg.get("contentRules") or []

    shift_regions = diff.shiftRegions or []
    shifted_height = sum(r.height for r in shift_regions)
    max_shift_band = max((r.height for r in shift_regions), default=0.0)
    net_shift = sum(r.height if r.type == "insertion" else -r.height for r in shift_regions)

    created = bundle.runContext.createdAt
    hour = created.hour + created.minute / 60.0
    hour_rad = 2 * math.pi * hour / 24.0

    ci = bundle.runContext.ci
    row: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "sampleId": bundle.sampleId,
        "projectId": bundle.projectId,
        "runId": bundle.runContext.runId,
        "createdAt": created.isoformat(),
        # G1
        "mismatch_pct_log": _log1p(diff.misMatchPercentage),
        "diff_pixel_ratio": diff.diffPixels / max(effective_total, 1.0),
        "ignored_ratio": (
            diff.ignoredPixels / diff.totalPixels
            if diff.ignoredPixels is not None and diff.totalPixels
            else float("nan")
        ),
        "page_area_log": _log1p(width * height),
        "page_aspect": height / max(width, 1.0),
        "height_delta_ratio": (height - baseline_height) / max(baseline_height, 1.0),
        # G2
        "region_count_log": spatial.region_count_log,
        "largest_region_area_ratio": spatial.largest_region_area_ratio,
        "top3_region_area_ratio": spatial.top3_region_area_ratio,
        "region_concentration": spatial.region_concentration,
        "centroid_y_rel": spatial.centroid_y_rel,
        "centroid_x_rel": spatial.centroid_x_rel,
        "bbox_coverage_w": spatial.bbox_coverage_w,
        "bbox_coverage_h": spatial.bbox_coverage_h,
        "touches_edge_count": spatial.touches_edge_count,
        "mean_region_compactness": spatial.mean_region_compactness,
        "isolated_pixel_ratio": spatial.isolated_pixel_ratio,
        "max_region_aspect": spatial.max_region_aspect,
        # G3
        "has_shift": _bool_float(diff.hasShift if diff.hasShift is not None else bool(shift_regions)),
        "shift_count_log": _log1p(len(shift_regions)),
        "shifted_height_ratio": shifted_height / max(height, 1.0),
        "max_shift_band_ratio": max_shift_band / max(height, 1.0),
        "net_shift_direction": net_shift / max(height, 1.0),
        # G4
        "elem_visual_change_count_log": _log1p(status_counts["visual_change"]),
        "elem_added_count_log": _log1p(status_counts["added"]),
        "elem_removed_count_log": _log1p(status_counts["removed"]),
        "elem_moved_count_log": _log1p(status_counts["moved"]),
        "elem_layout_change_count_log": _log1p(status_counts["layout_change"]),
        "max_elem_diff_pct": max_elem_pct,
        "interactive_elem_changed": interactive_changed if elements else float("nan"),
        "header_or_nav_affected": _section_affected(elements, {"nav", "header", "menu"}),
        "hero_affected": _section_affected(elements, {"hero", "banner", "jumbotron"}),
        "footer_affected": _section_affected(elements, {"footer"}),
        "carousel_affected": _section_affected(elements, {"carousel", "slider", "swiper"}),
        "min_match_confidence": min(confidences) if confidences else float("nan"),
        "mean_match_confidence": float(np.mean(confidences)) if confidences else float("nan"),
        "dynamic_region_diff_ratio": dynamic_diff_pixels / max(diff.diffPixels, 1.0),
        # G5
        "mismatch_threshold": float(scenario_cfg.get("misMatchThreshold", 0.1)),
        "n_comparison_rules_log": _log1p(len(comparison_rules)),
        "n_content_rules_log": _log1p(len(content_rules)),
        "stale_rules_count": float(len(diff.staleRules or [])),
        "requires_auth": _bool_float(scenario_cfg.get("requiresAuth")),
        # G6
        "viewport_class_ord": _ordinal(_viewport_class(width), VIEWPORT_CLASSES),
        "browser_ord": _ordinal(diff.browser, BROWSERS),
        "page_type_ord": _ordinal(bundle.runContext.pageType, VIEWPORT_CLASSES),
        "env_is_prod": _bool_float(bundle.runContext.envIsProd),
        "trigger_ord": _ordinal(bundle.runContext.trigger, TRIGGERS),
        "ci_present": _bool_float(ci is not None and ci.commitSha is not None),
        "ci_is_main_branch": _bool_float(ci.isMainBranch if ci else None),
        "ci_has_pr": _bool_float(ci.prNumber is not None if ci else None),
        "hour_sin": math.sin(hour_rad),
        "hour_cos": math.cos(hour_rad),
    }

    row.update(_historical_stats(bundle))
    row.update(_cross_view_features(bundle))
    row.update(_intent_github(bundle))
    row.update(_intent_jira(bundle))
    row.update(_intent_figma(bundle))
    row.update(_ai_features(bundle))

    if bundle.label and bundle.label.verdict:
        row[LABEL_COLUMN] = bundle.label.verdict

    for col in FEATURE_COLUMNS:
        row.setdefault(col, float("nan"))

    return row


def bundles_to_dataframe(bundles: list[DiffBundle]):
    import pandas as pd

    rows = [featurize_bundle(b) for b in bundles]
    return pd.DataFrame(rows)
