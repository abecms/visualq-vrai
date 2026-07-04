"""DiffBundle schema — versioned contract between VisualQ and visualq-vrai."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

BUNDLE_VERSION = 1
SCHEMA_VERSION = 1

TriClass = Literal[
    "intentional_redesign",
    "accidental_regression",
    "platform_constraint",
]

VerdictSource = Literal[
    "viewer",
    "slack",
    "mcp",
    "backfill-approve",
    "synthetic",
]


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class ShiftRegion(BaseModel):
    type: Literal["insertion", "deletion"]
    y: float
    height: float


class ElementDiffResult(BaseModel):
    selector: str
    tag: str
    ariaLabel: str | None = None
    semanticLabel: str | None = None
    bbox: BBox
    status: str
    mode: str
    diffPixelCount: float | None = None
    diffPercentage: float | None = None
    dimensionDelta: dict[str, float] | None = None
    positionDelta: dict[str, float] | None = None
    appliedRuleId: str | None = None
    matchConfidence: float | None = None


class BlockDiffResult(BaseModel):
    blockId: str
    selector: str
    title: str
    kind: str
    bbox: BBox
    status: str
    mode: str
    diffPixelCount: float
    diffPercentage: float
    childCount: int
    children: list[ElementDiffResult] = Field(default_factory=list)
    appliedRuleId: str | None = None


class AiDiffAnalysis(BaseModel):
    explanation: str
    severity: Literal["critical", "major", "minor", "cosmetic"]
    category: str
    confidence: float
    actionRecommended: Literal["approve", "investigate", "reject"]


class DiffResultRow(BaseModel):
    """Single VRT diff row — mirrors worker DiffResult."""

    scenario: str
    viewport: str
    browser: str
    status: Literal["pass", "fail"]
    misMatchPercentage: float
    diffPixels: float
    totalPixels: float
    dimensions: dict[str, float]
    hasContentRules: bool = False
    shiftRegions: list[ShiftRegion] | None = None
    hasShift: bool | None = None
    elementResults: list[ElementDiffResult] | None = None
    blockResults: list[BlockDiffResult] | None = None
    ignoredPixels: float | None = None
    effectiveTotalPixels: float | None = None
    staleRules: list[str] | None = None
    aiAnalysis: AiDiffAnalysis | None = None
    baselineHeight: float | None = None


class DOMMapNode(BaseModel):
    stableSelector: str
    tag: str
    bbox: BBox
    subtreeHash: str | None = None
    contentHash: str | None = None
    ariaRole: str | None = None
    ariaLabel: str | None = None
    text: str | None = None


class DOMMap(BaseModel):
    nodes: list[DOMMapNode]
    viewportSize: dict[str, float]
    pageHeight: float
    capturedAt: str | None = None


class BundleImages(BaseModel):
    baselinePng: str | None = None
    testPng: str | None = None
    diffPng: str | None = None
    baselineDomMap: str | None = None
    testDomMap: str | None = None


class CiContext(BaseModel):
    commitSha: str | None = None
    branch: str | None = None
    prNumber: int | None = None
    prTitle: str | None = None
    isMainBranch: bool | None = None
    jiraKey: str | None = None


class RunContext(BaseModel):
    runId: str
    createdAt: datetime
    trigger: str | None = None
    environmentId: str | None = None
    pageType: str | None = None
    envIsProd: bool | None = None
    ci: CiContext | None = None
    siblingResults: list[DiffResultRow] = Field(default_factory=list)
    scenarioConfig: dict[str, Any] | None = None


class GitHubIntentSignals(BaseModel):
    commitChangedSinceBaseline: bool | None = None
    prFilesChanged: int | None = None
    uiFilesChanged: bool | None = None
    cssLinesChanged: int | None = None
    prTitleType: str | None = None
    prHasDesignLabel: bool | None = None
    changedFilesMatchPage: bool | None = None


class JiraIntentSignals(BaseModel):
    jiraKeyPresent: bool | None = None
    jiraIssueIsStory: bool | None = None
    jiraIssueIsBug: bool | None = None
    jiraHasDesignLabel: bool | None = None


class FigmaIntentSignals(BaseModel):
    figmaDesignChangedRecently: bool | None = None
    daysSinceFigmaVersion: float | None = None
    figmaAlignmentBaseline: float | None = None
    figmaAlignmentDelta: float | None = None


class IntentSignals(BaseModel):
    github: GitHubIntentSignals | None = None
    jira: JiraIntentSignals | None = None
    figma: FigmaIntentSignals | None = None


class HistoricalRunSummary(BaseModel):
    runId: str
    createdAt: datetime
    misMatchPercentage: float
    status: str
    verdict: TriClass | None = None
    diffBBox: BBox | None = None


class DiffBundleLabel(BaseModel):
    verdict: TriClass | None = None
    verdictSource: VerdictSource | None = None
    verdictAt: datetime | None = None


class DiffBundle(BaseModel):
    """One diff sample — input to featurization and triage."""

    bundleVersion: Literal[1] = BUNDLE_VERSION
    sampleId: str
    orgId: str | None = None
    projectId: str | None = None
    scenarioId: str | None = None
    diff: DiffResultRow
    images: BundleImages = Field(default_factory=BundleImages)
    runContext: RunContext
    intentSignals: IntentSignals | None = None
    history: list[HistoricalRunSummary] = Field(default_factory=list)
    label: DiffBundleLabel | None = None

    def model_dump_json_schema(self) -> dict[str, Any]:
        return DiffBundle.model_json_schema()


class TriageClass(str, Enum):
    INTENTIONAL = "intentional_redesign"
    REGRESSION = "accidental_regression"
    PLATFORM = "platform_constraint"

    @classmethod
    def ordered(cls) -> list[TriageClass]:
        return [cls.INTENTIONAL, cls.REGRESSION, cls.PLATFORM]

    @classmethod
    def index(cls, value: str) -> int:
        mapping = {c.value: i for i, c in enumerate(cls.ordered())}
        if value not in mapping:
            raise ValueError(f"Unknown triage class: {value}")
        return mapping[value]
