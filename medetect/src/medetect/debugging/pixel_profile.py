"""Pixel profile helpers shared by sample scripts and tests."""

from __future__ import annotations

import numpy as np


def extract_line_profile(
    arr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel values sampled along the line from ``(x0, y0)`` to ``(x1, y1)``."""
    n = max(int(np.hypot(x1 - x0, y1 - y0)), 1)
    xs = np.round(np.linspace(x0, x1, n + 1)).astype(int)
    ys = np.round(np.linspace(y0, y1, n + 1)).astype(int)

    height, width = arr.shape[:2]
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    positions = np.linspace(0.0, float(np.hypot(x1 - x0, y1 - y0)), n + 1)
    values = arr[ys, xs]
    return positions, values


def normalize_values(values: np.ndarray) -> np.ndarray:
    """Normalize pixel values from [0, 255] to [0.0, 1.0]."""
    return values.astype(np.float32) / 255.0


def resolve_coords(
    args_coords: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert coordinates to absolute pixel indices.

    Values are treated as normalized when all four lie in [0, 1].
    Otherwise they are interpreted as absolute pixel indices.
    """
    x1, y1, x2, y2 = args_coords
    if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        px0 = int(round(x1 * (width - 1)))
        py0 = int(round(y1 * (height - 1)))
        px1 = int(round(x2 * (width - 1)))
        py1 = int(round(y2 * (height - 1)))
    else:
        px0, py0 = int(round(x1)), int(round(y1))
        px1, py1 = int(round(x2)), int(round(y2))
    return px0, py0, px1, py1
