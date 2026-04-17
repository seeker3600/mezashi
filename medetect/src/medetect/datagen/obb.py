"""OBB geometry and label formatting helpers for datagen."""

from __future__ import annotations

import math


def compute_obb_corners(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_rad: float,
) -> list[tuple[float, float]]:
    """Compute oriented bounding-box corners in clockwise order."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    hw, hh = w / 2, h / 2

    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in corners
    ]


def format_obb_label(
    class_id: int,
    corners: list[tuple[float, float]],
    img_w: int,
    img_h: int,
) -> str:
    """Format one YOLO OBB label line with normalised coordinates."""
    parts = [str(class_id)]
    for x, y in corners:
        parts.append(f"{x / img_w:.6f}")
        parts.append(f"{y / img_h:.6f}")
    return " ".join(parts)