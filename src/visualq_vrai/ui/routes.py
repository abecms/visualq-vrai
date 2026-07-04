"""FastAPI routes for the review & labeling UI."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from visualq_vrai.ui.store import BundleStore

DEFAULT_BUNDLES_DIR = "fixtures/bundles"

router = APIRouter()

_store: BundleStore | None = None


def get_store() -> BundleStore:
    global _store
    if _store is None:
        directory = os.environ.get("VRAI_BUNDLES_DIR", DEFAULT_BUNDLES_DIR)
        _store = BundleStore(directory)
    return _store


def reset_store() -> None:
    """Test hook — drop the singleton so the next request reloads."""
    global _store
    _store = None


class VerdictRequest(BaseModel):
    sampleId: str
    verdict: str


@router.get("/ui", response_class=HTMLResponse)
def ui_index() -> str:
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@router.get("/ui/api/bundles")
def list_bundles() -> dict:
    store = get_store()
    return {
        "bundles": store.summaries(),
        "labelStatus": store.label_status(),
        "tabicl": store.tabicl_meta,
    }


@router.get("/ui/api/bundles/{sample_id}")
def bundle_detail(sample_id: str) -> dict:
    store = get_store()
    try:
        return store.detail(sample_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown sampleId: {sample_id}")


@router.post("/ui/api/verdict")
def post_verdict(body: VerdictRequest) -> dict:
    store = get_store()
    try:
        return store.set_verdict(body.sampleId, body.verdict)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown sampleId: {body.sampleId}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/ui/api/refresh-tabicl")
def refresh_tabicl() -> dict:
    """Explicitly (re)compute TabICL triage for unlabeled bundles."""
    store = get_store()
    return store.refresh_tabicl()


@router.get("/ui/images/{sample_id}/{kind}")
def bundle_image(sample_id: str, kind: str) -> FileResponse:
    if kind not in {"baseline", "test", "diff"}:
        raise HTTPException(status_code=422, detail=f"Unknown image kind: {kind}")
    store = get_store()
    try:
        path = store.image_path(sample_id, kind)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown sampleId: {sample_id}")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(path, media_type="image/png")
