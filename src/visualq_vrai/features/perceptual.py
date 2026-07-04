"""Perceptual similarity between renders (design-as-oracle, G11).

Multi-scale pHash (DCT-based perceptual hash at several resolutions) — robust
to compression, anti-aliasing and small offsets, unlike pixel comparison.
Returns a similarity in [0, 1]; 1.0 means perceptually identical.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.fft import dctn

_SCALES = (32, 16, 8)
_HASH_KEEP = {32: 8, 16: 6, 8: 4}


def _phash_bits(gray: np.ndarray, size: int) -> np.ndarray:
    """Perceptual hash bits of a grayscale image resized to size×size."""
    img = Image.fromarray(gray).resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)
    coeffs = dctn(arr, norm="ortho")
    keep = _HASH_KEEP[size]
    low = coeffs[:keep, :keep].flatten()
    # Exclude the DC term from the median so uniform images hash stably.
    median = np.median(low[1:])
    return low > median


def _to_grayscale(image: Image.Image) -> np.ndarray:
    # Flatten alpha on white: Figma renders come with transparent backgrounds.
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image.convert("RGBA"))
    return np.asarray(image.convert("L"))


def perceptual_similarity(image_a: bytes, image_b: bytes) -> float:
    """Multi-scale pHash similarity between two PNG/JPEG byte payloads."""
    gray_a = _to_grayscale(Image.open(io.BytesIO(image_a)))
    gray_b = _to_grayscale(Image.open(io.BytesIO(image_b)))

    similarities: list[float] = []
    for size in _SCALES:
        bits_a = _phash_bits(gray_a, size)
        bits_b = _phash_bits(gray_b, size)
        hamming = float(np.count_nonzero(bits_a != bits_b))
        similarities.append(1.0 - hamming / bits_a.size)

    return float(np.mean(similarities))
