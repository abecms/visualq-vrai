"""Feature column specification — schemaVersion 1, 86 dense columns (G1–G12)."""

from __future__ import annotations

SCHEMA_VERSION = 1
MIN_CONTEXT_ROWS = 300
MIN_CLASS_EXAMPLES = 15

# Ordered feature names used in Parquet I/O and TabICL fit/predict.
FEATURE_COLUMNS: list[str] = [
    # G1 — magnitude (6)
    "mismatch_pct_log",
    "diff_pixel_ratio",
    "ignored_ratio",
    "page_area_log",
    "page_aspect",
    "height_delta_ratio",
    # G2 — spatial distribution (12)
    "region_count_log",
    "largest_region_area_ratio",
    "top3_region_area_ratio",
    "region_concentration",
    "centroid_y_rel",
    "centroid_x_rel",
    "bbox_coverage_w",
    "bbox_coverage_h",
    "touches_edge_count",
    "mean_region_compactness",
    "isolated_pixel_ratio",
    "max_region_aspect",
    # G3 — shift (5)
    "has_shift",
    "shift_count_log",
    "shifted_height_ratio",
    "max_shift_band_ratio",
    "net_shift_direction",
    # G4 — DOM semantics (14)
    "elem_visual_change_count_log",
    "elem_added_count_log",
    "elem_removed_count_log",
    "elem_moved_count_log",
    "elem_layout_change_count_log",
    "max_elem_diff_pct",
    "interactive_elem_changed",
    "header_or_nav_affected",
    "hero_affected",
    "footer_affected",
    "carousel_affected",
    "min_match_confidence",
    "mean_match_confidence",
    "dynamic_region_diff_ratio",
    # G5 — configuration (5)
    "mismatch_threshold",
    "n_comparison_rules_log",
    "n_content_rules_log",
    "stale_rules_count",
    "requires_auth",
    # G6 — execution context (10) — categoricals encoded as float ordinals at serve time
    "viewport_class_ord",
    "browser_ord",
    "page_type_ord",
    "env_is_prod",
    "trigger_ord",
    "ci_present",
    "ci_is_main_branch",
    "ci_has_pr",
    "hour_sin",
    "hour_cos",
    # G7 — historical as-of (10)
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
    # G8 — cross-view consistency (5)
    "viewport_fail_fraction",
    "browser_fail_fraction",
    "only_this_browser",
    "run_fail_fraction",
    "same_zone_cross_page_fraction",
    # G9 — GitHub intent (7)
    "commit_changed_since_baseline",
    "pr_files_changed_log",
    "ui_files_changed",
    "css_lines_changed_log",
    "pr_title_type_ord",
    "pr_has_design_label",
    "changed_files_match_page",
    # G10 — JIRA intent (4)
    "jira_key_present",
    "jira_issue_is_story",
    "jira_issue_is_bug",
    "jira_has_design_label",
    # G11 — Figma intent (4)
    "figma_design_changed_recently",
    "days_since_figma_version_log",
    "figma_alignment_baseline",
    "figma_alignment_delta",
    # G12 — vision distilled (4)
    "ai_severity_ord",
    "ai_category_ord",
    "ai_confidence",
    "ai_action_ord",
]

LABEL_COLUMN = "label_class"

VIEWPORT_CLASSES = ["mobile", "tablet", "desktop", "wide", "unknown"]
BROWSERS = ["chromium", "firefox", "webkit", "unknown"]
TRIGGERS = ["manual", "ci", "schedule", "api", "unknown"]
PR_TITLE_TYPES = ["feat", "fix", "style", "refactor", "chore", "other"]
AI_CATEGORIES = [
    "layout",
    "content",
    "color",
    "typography",
    "spacing",
    "component",
    "animation",
    "third_party",
    "unknown",
    "other",
]

assert len(FEATURE_COLUMNS) == 86
