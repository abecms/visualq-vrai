"""Connector ablation mechanics.

Intent: ablation must remove exactly one connector's evidence (block NaN,
matching a project that never configured it) and nothing else — that is what
makes the measured contribution attributable to the connector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visualq_vrai.eval.harness import INTENT_GROUPS, ablate_intent_group
from visualq_vrai.schema.feature_spec import FEATURE_COLUMNS


def test_intent_groups_cover_expected_columns():
    all_intent = {c for cols in INTENT_GROUPS.values() for c in cols}
    assert all_intent <= set(FEATURE_COLUMNS)
    assert len(INTENT_GROUPS["github"]) == 7
    assert len(INTENT_GROUPS["jira"]) == 4
    assert len(INTENT_GROUPS["figma"]) == 4


def test_ablation_nans_only_target_group():
    df = pd.DataFrame([{c: 1.0 for c in FEATURE_COLUMNS}])
    out = ablate_intent_group(df, "figma")
    for col in INTENT_GROUPS["figma"]:
        assert np.isnan(out[col].iloc[0])
    for col in INTENT_GROUPS["github"] + INTENT_GROUPS["jira"]:
        assert out[col].iloc[0] == 1.0
    # Original untouched.
    assert df[INTENT_GROUPS["figma"][0]].iloc[0] == 1.0
