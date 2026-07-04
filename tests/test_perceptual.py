"""Perceptual similarity (design-as-oracle) — pHash multi-scale."""

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from visualq_vrai.features.perceptual import perceptual_similarity
from visualq_vrai.service.app import app


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _layout(shift_header: int = 0, jpeg: bool = False) -> bytes:
    """Synthetic page layout: header band, sidebar, content blocks."""
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, shift_header, 400, 40 + shift_header], fill=(30, 60, 200))
    draw.rectangle([0, 60, 80, 300], fill=(240, 240, 240))
    for i in range(4):
        draw.rectangle([100, 70 + i * 55, 380, 110 + i * 55], fill=(200, 210, 220))
    if jpeg:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    return _png_bytes(img)


def test_identical_images_are_maximally_similar():
    assert perceptual_similarity(_layout(), _layout()) == 1.0


def test_compression_noise_does_not_break_alignment():
    # JPEG artifacts must not look like a design change — that is exactly why
    # pixelmatch is the wrong tool for design-as-oracle.
    assert perceptual_similarity(_layout(), _layout(jpeg=True)) > 0.9


def test_layout_change_lowers_similarity():
    moved = _layout(shift_header=120)
    aligned = perceptual_similarity(_layout(), _layout(jpeg=True))
    changed = perceptual_similarity(_layout(), moved)
    assert changed < aligned


def test_transparent_background_is_flattened_white():
    rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    white = Image.new("RGB", (100, 100), "white")
    score = perceptual_similarity(_png_bytes(rgba), _png_bytes(white))
    assert score == 1.0


def test_similarity_endpoint_roundtrip():
    client = TestClient(app)
    res = client.post("/similarity", json={
        "imageA": base64.b64encode(_layout()).decode(),
        "imageB": base64.b64encode(_layout(jpeg=True)).decode(),
    })
    assert res.status_code == 200
    assert res.json()["similarity"] > 0.9


def test_similarity_endpoint_fails_loud_on_garbage():
    client = TestClient(app)
    res = client.post("/similarity", json={"imageA": "bm90LWFuLWltYWdl", "imageB": "bm90LWFuLWltYWdl"})
    assert res.status_code == 422
