"""Hull profile geometry for synthetic ship generation.

All coordinates are normalised: beam direction x ∈ [0, 1],
length direction y ∈ [0, lb_ratio].  Bow is at y = 0.
"""

from __future__ import annotations

import random

import numpy as np
from numpy.typing import NDArray

# ── Hull profile control points ──────────────────────────────────────────
# Each profile: list of (position_from_bow ∈ [0,1], half_width ∈ [0,0.5]).

PROFILES: dict[str, list[tuple[float, float]]] = {
    "warship": [
        (0.00, 0.00),
        (0.08, 0.25),
        (0.20, 0.45),
        (0.45, 0.50),
        (0.75, 0.42),
        (0.92, 0.25),
        (1.00, 0.15),
    ],
    "carrier": [
        (0.00, 0.10),
        (0.05, 0.35),
        (0.10, 0.48),
        (0.50, 0.50),
        (0.90, 0.48),
        (0.95, 0.40),
        (1.00, 0.30),
    ],
    "box": [
        (0.00, 0.15),
        (0.08, 0.40),
        (0.15, 0.48),
        (0.50, 0.50),
        (0.85, 0.48),
        (0.95, 0.42),
        (1.00, 0.35),
    ],
    "fishing": [
        (0.00, 0.00),
        (0.10, 0.30),
        (0.25, 0.45),
        (0.50, 0.50),
        (0.75, 0.45),
        (0.90, 0.35),
        (1.00, 0.10),
    ],
    "fishing_wide": [
        (0.00, 0.05),
        (0.10, 0.35),
        (0.20, 0.47),
        (0.50, 0.50),
        (0.75, 0.47),
        (0.88, 0.38),
        (1.00, 0.15),
    ],
    # Slender modern warship — tumblehome / stealth hull
    "warship_lean": [
        (0.00, 0.00),
        (0.10, 0.18),
        (0.22, 0.40),
        (0.45, 0.46),
        (0.70, 0.42),
        (0.88, 0.25),
        (1.00, 0.12),
    ],
    # Full-bodied tanker / supply ship
    "tanker": [
        (0.00, 0.10),
        (0.06, 0.32),
        (0.14, 0.48),
        (0.50, 0.50),
        (0.86, 0.48),
        (0.94, 0.40),
        (1.00, 0.30),
    ],
    # Tugboat — very blunt bow, wide midsection, rounded stern
    "tug": [
        (0.00, 0.12),
        (0.08, 0.38),
        (0.18, 0.48),
        (0.45, 0.50),
        (0.70, 0.50),
        (0.88, 0.42),
        (1.00, 0.25),
    ],
    # Barge / flat-bottom — nearly rectangular
    "barge": [
        (0.00, 0.30),
        (0.05, 0.46),
        (0.10, 0.50),
        (0.50, 0.50),
        (0.90, 0.50),
        (0.95, 0.46),
        (1.00, 0.30),
    ],
    # Small workboat — wide and blunt
    "workboat": [
        (0.00, 0.08),
        (0.10, 0.38),
        (0.22, 0.48),
        (0.50, 0.50),
        (0.78, 0.48),
        (0.90, 0.40),
        (1.00, 0.18),
    ],
    # Pilot / patrol launch — moderate V-bow, wide stern
    "launch": [
        (0.00, 0.00),
        (0.10, 0.28),
        (0.22, 0.44),
        (0.50, 0.50),
        (0.78, 0.50),
        (0.90, 0.44),
        (1.00, 0.28),
    ],
}


def interpolate_hull(
    profile_key: str,
    bow_sharpness: float,
    stern_hw: float,
    n_points: int = 64,
) -> NDArray[np.float64]:
    """Interpolate hull profile to *n_points* half-width values.

    Parameters
    ----------
    profile_key
        Key into :data:`PROFILES`.
    bow_sharpness
        0 = blunt bow, 1 = sharp bow.  Modulates width near the bow.
    stern_hw
        Half-width at the stern tip (replaces last control point).
    n_points
        Number of sample points along the length.

    Returns
    -------
    NDArray[np.float64]
        Half-width at each sample, values in [0, 0.5].
    """
    pts = list(PROFILES[profile_key])
    for i, (pos, hw) in enumerate(pts):
        if 0 < pos <= 0.15:
            factor = 0.5 + 0.5 * (1.0 - bow_sharpness)
            pts[i] = (pos, hw * factor)
    pts[-1] = (pts[-1][0], stern_hw)

    xs = np.array([p[0] for p in pts])
    hws = np.array([p[1] for p in pts])
    t = np.linspace(0.0, 1.0, n_points)
    return np.interp(t, xs, hws)


def apply_hull_trait_variant(
    half_widths: NDArray[np.float64],
    variant: object | None,
) -> NDArray[np.float64]:
    """Apply optional hull-shape traits to an interpolated half-width profile."""
    hw = np.array(half_widths, dtype=np.float64, copy=True)
    if hw.size == 0 or variant is None:
        return hw

    if bool(getattr(variant, "pointed_bow", False)):
        bow_end = max(3, int(round((hw.size - 1) * 0.22)))
        bow_scale = np.linspace(0.55, 1.0, bow_end + 1)
        hw[:bow_end + 1] *= bow_scale

    if bool(getattr(variant, "long_foredeck", False)):
        foredeck_end = max(5, int(round((hw.size - 1) * 0.48)))
        taper = np.linspace(0.0, 1.0, foredeck_end + 1)
        foredeck_scale = 0.36 + 0.64 * np.power(taper, 0.60)
        hw[:foredeck_end + 1] *= foredeck_scale

    if bool(getattr(variant, "straight_sides", False)):
        start = max(1, int(round((hw.size - 1) * 0.24)))
        end = max(start + 2, int(round((hw.size - 1) * 0.82)))
        mid = hw[start:end + 1]
        plateau = float(np.quantile(mid, 0.88))
        hw[start:end + 1] = np.clip(0.25 * mid + 0.75 * plateau, 0.0, 0.5)

    round_stern_active = bool(getattr(variant, "round_stern", False))

    if bool(getattr(variant, "square_stern", False)) and not round_stern_active:
        stern_start = max(1, int(round((hw.size - 1) * 0.82)))
        aft_ref_start = max(0, int(round((hw.size - 1) * 0.55)))
        aft_ref_end = max(aft_ref_start + 1, int(round((hw.size - 1) * 0.78)))
        aft_ref = float(np.quantile(hw[aft_ref_start:aft_ref_end + 1], 0.80))
        target = min(0.5, max(hw[-1] + 0.08, aft_ref * 0.92))
        ramp_start = max(hw[stern_start], target * 0.94)
        ramp = np.linspace(ramp_start, target, hw.size - stern_start)
        hw[stern_start:] = np.maximum(hw[stern_start:], ramp)
        hw[-1] = max(hw[-1], target)

    if round_stern_active:
        # Quarter-ellipse curve from ~72% LOA to stern tip.
        # Creates a smooth rounded (cruiser) stern shape.
        stern_start = max(1, int(round((hw.size - 1) * 0.72)))
        region_len = hw.size - stern_start
        if region_len >= 3:
            max_w = float(hw[stern_start])
            stern_tip_w = float(hw[-1])
            t_vals = np.linspace(0.0, 1.0, region_len)
            alpha = np.sqrt(np.maximum(0.0, 1.0 - t_vals ** 2))
            hw[stern_start:] = stern_tip_w + (max_w - stern_tip_w) * alpha

    return np.clip(hw, 0.0, 0.5)


def build_hull_points(
    half_widths: NDArray[np.float64],
    lb_ratio: float,
    rng: random.Random,
    noise_scale: float = 0.005,
) -> list[tuple[float, float]]:
    """Build hull polygon in normalised SVG coordinates.

    Returns vertices traced clockwise: starboard side bow→stern,
    then port side stern→bow.

    Coordinates: x ∈ [0, 1] (beam), y ∈ [0, lb_ratio] (length).
    """
    n = len(half_widths)
    noise = np.array([rng.gauss(0, noise_scale) for _ in range(n)])
    hw = np.clip(half_widths + noise, 0.0, 0.5)

    right: list[tuple[float, float]] = []
    left: list[tuple[float, float]] = []
    for i, w in enumerate(hw):
        t = i / max(n - 1, 1)
        y = t * lb_ratio
        right.append((0.5 + float(w), y))
        left.append((0.5 - float(w), y))
    return right + list(reversed(left))
