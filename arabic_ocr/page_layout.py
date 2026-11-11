"""Projection-based line candidates for clean, single-column printed pages.

Coordinates refer to the EXIF-oriented image. This is a heuristic, not a layout
detector: tables, columns, heavy skew and touching lines require another method.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("L")


def otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    weight = histogram.cumsum()
    moment = (histogram * np.arange(256)).cumsum()
    denominator = weight * (weight[-1] - weight)
    score = np.zeros(256)
    valid = denominator > 0
    score[valid] = (moment[-1] * weight[valid] - moment[valid] * weight[-1]) ** 2 / denominator[valid]
    return int(np.clip(score.argmax(), 40, 220))


def detect_line_boxes(image: Image.Image, *, max_gap: int = 2, padding: int = 4) -> list[tuple[int, int, int, int]]:
    if max_gap < 0 or padding < 0:
        raise ValueError("max_gap and padding must be non-negative")
    gray = np.asarray(prepare_image(image))
    ink = gray <= otsu_threshold(gray)
    row_ink = ink.sum(axis=1)
    active = row_ink >= max(3, round(gray.shape[1] * 0.002))
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > max_gap + 1) + 1
    bands = [(int(group[0]), int(group[-1]) + 1) for group in np.split(indices, cuts)]
    # Attach small diacritic bands to a nearby body, preserving Arabic dots.
    bodies = [bottom - top for top, bottom in bands if bottom - top >= 5]
    body_height = float(np.median(bodies)) if bodies else 0
    if body_height:
        while len(bands) > 1:
            candidates = []
            for i in range(len(bands) - 1):
                first, second = bands[i:i + 2]
                gap = second[0] - first[1]
                small = min(first[1] - first[0], second[1] - second[0]) <= max(2, body_height * 0.3)
                if small and gap <= max(3, body_height * 0.65):
                    candidates.append((gap, i))
            if not candidates:
                break
            _, i = min(candidates)
            bands[i:i + 2] = [(bands[i][0], bands[i + 1][1])]
    boxes = []
    for i, (top, bottom) in enumerate(bands):
        if bottom - top < 2:
            continue
        cols = np.flatnonzero(ink[top:bottom].sum(axis=0) >= 2)
        if not len(cols):
            continue
        lower = 0 if i == 0 else (bands[i - 1][1] + top) // 2
        upper = gray.shape[0] if i + 1 == len(bands) else (bottom + bands[i + 1][0]) // 2
        boxes.append((max(0, int(cols[0]) - padding), max(lower, top - padding),
                      min(gray.shape[1], int(cols[-1]) + 1 + padding), min(upper, bottom + padding)))
    return boxes
