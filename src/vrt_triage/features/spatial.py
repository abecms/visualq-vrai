"""Spatial diff features (G2) from diff PNG via connected components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class SpatialFeatures:
    region_count_log: float
    largest_region_area_ratio: float
    top3_region_area_ratio: float
    region_concentration: float
    centroid_y_rel: float
    centroid_x_rel: float
    bbox_coverage_w: float
    bbox_coverage_h: float
    touches_edge_count: float
    mean_region_compactness: float
    isolated_pixel_ratio: float
    max_region_aspect: float


def _load_diff_mask(path: str | Path, width: int, height: int) -> np.ndarray:
    """Return boolean mask of diff pixels (red channel in VisualQ diff PNG)."""
    img = Image.open(path).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.NEAREST)
    arr = np.array(img)
    red = arr[:, :, 0] == 255
    green = arr[:, :, 1] == 0
    blue = arr[:, :, 2] == 0
    alpha = arr[:, :, 3] > 0
    return red & green & blue & alpha


def _nan_spatial() -> SpatialFeatures:
    nan = float("nan")
    return SpatialFeatures(
        region_count_log=nan,
        largest_region_area_ratio=nan,
        top3_region_area_ratio=nan,
        region_concentration=nan,
        centroid_y_rel=nan,
        centroid_x_rel=nan,
        bbox_coverage_w=nan,
        bbox_coverage_h=nan,
        touches_edge_count=nan,
        mean_region_compactness=nan,
        isolated_pixel_ratio=nan,
        max_region_aspect=nan,
    )


def extract_spatial_features(
    diff_png: str | None,
    width: int,
    height: int,
    *,
    dilation_radius: int = 5,
    min_pixels: int = 10,
) -> SpatialFeatures:
    if not diff_png or not Path(diff_png).is_file() or width <= 0 or height <= 0:
        return _nan_spatial()

    mask = _load_diff_mask(diff_png, width, height)
    total_pixels = width * height
    diff_count = int(mask.sum())
    if diff_count == 0:
        return SpatialFeatures(
            region_count_log=0.0,
            largest_region_area_ratio=0.0,
            top3_region_area_ratio=0.0,
            region_concentration=0.0,
            centroid_y_rel=float("nan"),
            centroid_x_rel=float("nan"),
            bbox_coverage_w=0.0,
            bbox_coverage_h=0.0,
            touches_edge_count=0.0,
            mean_region_compactness=float("nan"),
            isolated_pixel_ratio=0.0,
            max_region_aspect=float("nan"),
        )

    if dilation_radius > 0:
        structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
        mask = ndimage.binary_dilation(mask, structure=structure)

    labeled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return _nan_spatial()

    sizes = ndimage.sum(mask, labeled, index=range(1, n_labels + 1))
    regions: list[dict[str, float]] = []
    isolated_pixels = 0

    for label_idx in range(1, n_labels + 1):
        region_mask = labeled == label_idx
        count = int(region_mask.sum())
        if count < min_pixels:
            if count == 1:
                isolated_pixels += 1
            continue
        ys, xs = np.where(region_mask)
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        bw = max_x - min_x + 1
        bh = max_y - min_y + 1
        compactness = count / max(bw * bh, 1)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        touches = 0
        if min_x == 0:
            touches += 1
        if max_x == width - 1:
            touches += 1
        if min_y == 0:
            touches += 1
        if max_y == height - 1:
            touches += 1
        regions.append(
            {
                "count": float(count),
                "min_x": float(min_x),
                "max_x": float(max_x),
                "min_y": float(min_y),
                "max_y": float(max_y),
                "compactness": float(compactness),
                "aspect": float(aspect),
                "touches": float(touches),
                "cx": float(xs.mean()),
                "cy": float(ys.mean()),
            }
        )

    if not regions:
        return SpatialFeatures(
            region_count_log=math.log1p(0),
            largest_region_area_ratio=0.0,
            top3_region_area_ratio=0.0,
            region_concentration=0.0,
            centroid_y_rel=float("nan"),
            centroid_x_rel=float("nan"),
            bbox_coverage_w=0.0,
            bbox_coverage_h=0.0,
            touches_edge_count=0.0,
            mean_region_compactness=float("nan"),
            isolated_pixel_ratio=isolated_pixels / max(diff_count, 1),
            max_region_aspect=float("nan"),
        )

    regions.sort(key=lambda r: r["count"], reverse=True)
    area_ratios = [r["count"] / total_pixels for r in regions]
    largest = area_ratios[0]
    top3 = sum(area_ratios[:3])
    concentration = largest / max(sum(area_ratios), 1e-9)

    union_min_x = min(r["min_x"] for r in regions)
    union_max_x = max(r["max_x"] for r in regions)
    union_min_y = min(r["min_y"] for r in regions)
    union_max_y = max(r["max_y"] for r in regions)
    bbox_w = (union_max_x - union_min_x + 1) / width
    bbox_h = (union_max_y - union_min_y + 1) / height

    weighted_cx = sum(r["cx"] * r["count"] for r in regions) / sum(r["count"] for r in regions)
    weighted_cy = sum(r["cy"] * r["count"] for r in regions) / sum(r["count"] for r in regions)

    return SpatialFeatures(
        region_count_log=math.log1p(len(regions)),
        largest_region_area_ratio=largest,
        top3_region_area_ratio=top3,
        region_concentration=concentration,
        centroid_y_rel=weighted_cy / height,
        centroid_x_rel=weighted_cx / width,
        bbox_coverage_w=bbox_w,
        bbox_coverage_h=bbox_h,
        touches_edge_count=max(r["touches"] for r in regions),
        mean_region_compactness=float(np.mean([r["compactness"] for r in regions])),
        isolated_pixel_ratio=isolated_pixels / max(diff_count, 1),
        max_region_aspect=max(r["aspect"] for r in regions),
    )
