"""Review UI tests — verdict persistence in bundle files, sorting, insufficient-data state.

Intent: the bundle JSON on disk is the single source of truth for labels
(reintegration contract with VisualQ), the list is ranked by regression
probability so reviewers see risky diffs first, and the UI must never present
heuristic output as model output (explicit source + insufficient-data banner).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visualq_vrai.ui import routes
from visualq_vrai.ui.store import BundleStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "bundles"


@pytest.fixture()
def bundle_dir(tmp_path: Path) -> Path:
    target = tmp_path / "bundles"
    target.mkdir()
    for src in sorted(FIXTURES.glob("*.json"))[:12]:
        shutil.copy(src, target / src.name)
    return target


@pytest.fixture()
def client(bundle_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VRAI_BUNDLES_DIR", str(bundle_dir))
    routes.reset_store()
    from visualq_vrai.service.app import app

    with TestClient(app) as test_client:
        yield test_client
    routes.reset_store()


def _first_sample_id(bundle_dir: Path) -> str:
    first = sorted(bundle_dir.glob("*.json"))[0]
    return json.loads(first.read_text())["sampleId"]


def test_verdict_written_to_bundle_file(client: TestClient, bundle_dir: Path):
    sample_id = _first_sample_id(bundle_dir)
    resp = client.post(
        "/ui/api/verdict",
        json={"sampleId": sample_id, "verdict": "platform_constraint"},
    )
    assert resp.status_code == 200

    path = next(
        p for p in bundle_dir.glob("*.json")
        if json.loads(p.read_text())["sampleId"] == sample_id
    )
    on_disk = json.loads(path.read_text())
    assert on_disk["label"]["verdict"] == "platform_constraint"
    assert on_disk["label"]["verdictSource"] == "viewer"
    assert on_disk["label"]["verdictAt"] is not None


def test_all_three_verdict_classes_accepted(client: TestClient, bundle_dir: Path):
    ids = [json.loads(p.read_text())["sampleId"] for p in sorted(bundle_dir.glob("*.json"))[:3]]
    verdicts = ["intentional_redesign", "accidental_regression", "platform_constraint"]
    for sample_id, verdict in zip(ids, verdicts):
        resp = client.post("/ui/api/verdict", json={"sampleId": sample_id, "verdict": verdict})
        assert resp.status_code == 200
        assert resp.json()["verdict"] == verdict


def test_unknown_verdict_rejected(client: TestClient, bundle_dir: Path):
    sample_id = _first_sample_id(bundle_dir)
    resp = client.post("/ui/api/verdict", json={"sampleId": sample_id, "verdict": "looks_fine"})
    assert resp.status_code == 422


def test_unknown_sample_rejected(client: TestClient):
    resp = client.post(
        "/ui/api/verdict",
        json={"sampleId": "does-not-exist", "verdict": "accidental_regression"},
    )
    assert resp.status_code == 404


def test_list_sorted_by_regression_probability(client: TestClient):
    data = client.get("/ui/api/bundles").json()
    probs = [b["regressionProbability"] for b in data["bundles"]]
    assert probs == sorted(probs, reverse=True)
    for b in data["bundles"]:
        assert b["source"] in {"heuristic", "tabicl"}


def test_insufficient_data_state_exposed(client: TestClient):
    """12 bundles << 300 — the model must be reported as unavailable, not faked."""
    data = client.get("/ui/api/bundles").json()
    status = data["labelStatus"]
    assert status["insufficientData"] is True
    assert status["remainingToMinContext"] > 0
    # All triage must be heuristic in this regime.
    assert all(b["source"] == "heuristic" for b in data["bundles"])


def test_verdict_invalidates_tabicl_cache(bundle_dir: Path):
    store = BundleStore(bundle_dir)
    store._tabicl_cache = {"fake": {"class": "accidental_regression"}}
    store._tabicl_meta = {"state": "ready"}
    sample_id = _first_sample_id(bundle_dir)
    store.set_verdict(sample_id, "accidental_regression")
    assert store._tabicl_cache == {}
    assert store.tabicl_meta["state"] == "stale"


def test_detail_shape(client: TestClient, bundle_dir: Path):
    sample_id = _first_sample_id(bundle_dir)
    detail = client.get(f"/ui/api/bundles/{sample_id}").json()
    assert detail["sampleId"] == sample_id
    assert detail["triage"]["source"] == "heuristic"
    assert set(detail["images"].keys()) == {"baseline", "test", "diff"}


def test_ui_page_served(client: TestClient):
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "VRAI" in resp.text
