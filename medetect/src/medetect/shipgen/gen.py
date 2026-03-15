"""Synthetic ship silhouette generator using shape grammar.

Generates transparent RGBA PNG images of ship top-down silhouettes for
object detection training.  Ships are composed from geometric primitives:
hull polygon + superstructure rectangles + detail marks.

Ship classes and their parameters are derived from the design documents
(docs/軍用艦艇.md, docs/特殊艦艇など.md).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ── Hull profile control points ──────────────────────────────────────────
# Each profile: list of (position_from_bow ∈ [0,1], half_width ∈ [0,0.5]).
# Interpolated linearly to produce the hull outline.

_PROFILES: dict[str, list[tuple[float, float]]] = {
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
}

# ── Superstructure / detail rule data classes ────────────────────────────


@dataclass(frozen=True)
class _Struct:
    """Superstructure block placement rule.

    Positions are normalised: bow = 0, stern = 1 along ship length;
    0 = port edge, 1 = starboard edge across beam.
    """

    x0: tuple[float, float]  # start position range along length
    x1: tuple[float, float]  # end position range along length
    w: tuple[float, float]  # width as fraction of beam
    y_off: float = 0.0  # lateral offset from centre (-0.5..0.5)
    shade_off: int = 30  # brightness offset from hull base colour
    prob: float = 1.0  # probability of placement


@dataclass(frozen=True)
class _Detail:
    """Small detail element placement rule.

    *kind* selects the drawing routine; *x* is along ship length,
    *y* is across beam (0.5 = centre).
    """

    kind: str  # mast|gun|helipad|circle_spot|vls|crane|lamp|line|door|elevator
    x: tuple[float, float]  # position range along length
    y: float = 0.5  # across beam (0 = port, 1 = starboard)
    size: float = 0.03  # relative to ship length
    prob: float = 1.0


@dataclass(frozen=True)
class _ShipClass:
    """Complete ship class template."""

    hull: str  # key into _PROFILES
    lb: tuple[float, float]  # length / beam ratio range
    bow: tuple[float, float]  # bow sharpness (0 = blunt, 1 = sharp)
    stern_hw: tuple[float, float]  # stern half-width range
    shade: tuple[int, int]  # hull base gray value range
    structs: tuple[_Struct, ...]
    details: tuple[_Detail, ...]


# ── Ship class registry ──────────────────────────────────────────────────

SHIP_CLASSES: dict[str, _ShipClass] = {
    # --- Military: normal warships ---
    "patrol": _ShipClass(
        hull="warship",
        lb=(6.0, 9.0),
        bow=(0.6, 0.9),
        stern_hw=(0.05, 0.15),
        shade=(120, 160),
        structs=(
            _Struct(x0=(0.22, 0.28), x1=(0.40, 0.48), w=(0.35, 0.55)),
        ),
        details=(
            _Detail("mast", x=(0.38, 0.45), size=0.04),
            _Detail("gun", x=(0.08, 0.14), size=0.03),
        ),
    ),
    "corvette": _ShipClass(
        hull="warship",
        lb=(7.5, 9.0),
        bow=(0.6, 0.85),
        stern_hw=(0.10, 0.20),
        shade=(120, 160),
        structs=(
            _Struct(x0=(0.22, 0.28), x1=(0.38, 0.44), w=(0.35, 0.60)),
            _Struct(
                x0=(0.48, 0.55), x1=(0.62, 0.68), w=(0.30, 0.50), shade_off=20,
            ),
        ),
        details=(
            _Detail("mast", x=(0.40, 0.48), size=0.04),
            _Detail("mast", x=(0.62, 0.68), size=0.03, prob=0.6),
            _Detail("gun", x=(0.08, 0.14), size=0.03),
            _Detail("helipad", x=(0.80, 0.90), size=0.08, prob=0.5),
        ),
    ),
    "frigate": _ShipClass(
        hull="warship",
        lb=(8.0, 10.0),
        bow=(0.7, 0.95),
        stern_hw=(0.12, 0.20),
        shade=(120, 155),
        structs=(
            _Struct(x0=(0.22, 0.28), x1=(0.38, 0.44), w=(0.40, 0.65)),
            _Struct(
                x0=(0.46, 0.52), x1=(0.58, 0.64), w=(0.35, 0.55), shade_off=25,
            ),
            _Struct(
                x0=(0.64, 0.70),
                x1=(0.72, 0.78),
                w=(0.30, 0.50),
                shade_off=20,
                prob=0.7,
            ),
        ),
        details=(
            _Detail("mast", x=(0.40, 0.46), size=0.04),
            _Detail("mast", x=(0.60, 0.66), size=0.035, prob=0.8),
            _Detail("gun", x=(0.08, 0.14), size=0.03),
            _Detail("vls", x=(0.15, 0.20), size=0.04, prob=0.6),
            _Detail("helipad", x=(0.82, 0.92), size=0.08),
        ),
    ),
    "destroyer": _ShipClass(
        hull="warship",
        lb=(8.5, 10.0),
        bow=(0.75, 0.95),
        stern_hw=(0.12, 0.20),
        shade=(115, 150),
        structs=(
            _Struct(x0=(0.20, 0.26), x1=(0.36, 0.42), w=(0.45, 0.70)),
            _Struct(
                x0=(0.44, 0.50), x1=(0.56, 0.62), w=(0.40, 0.60), shade_off=25,
            ),
            _Struct(
                x0=(0.62, 0.68), x1=(0.72, 0.78), w=(0.35, 0.55), shade_off=20,
            ),
        ),
        details=(
            _Detail("mast", x=(0.38, 0.44), size=0.045),
            _Detail("mast", x=(0.58, 0.64), size=0.04),
            _Detail("gun", x=(0.07, 0.12), size=0.035),
            _Detail("vls", x=(0.14, 0.19), size=0.05),
            _Detail("vls", x=(0.42, 0.46), size=0.04, prob=0.5),
            _Detail("helipad", x=(0.82, 0.92), size=0.09),
        ),
    ),
    # --- Military: deck-dominated ---
    "carrier": _ShipClass(
        hull="carrier",
        lb=(7.0, 9.0),
        bow=(0.3, 0.5),
        stern_hw=(0.25, 0.35),
        shade=(130, 160),
        structs=(
            _Struct(
                x0=(0.35, 0.45),
                x1=(0.55, 0.65),
                w=(0.10, 0.18),
                y_off=0.30,
                shade_off=35,
            ),
        ),
        details=(
            _Detail("line", x=(0.30, 0.40), y=0.3, size=0.50, prob=0.7),
            _Detail("elevator", x=(0.25, 0.30), y=0.85, size=0.06, prob=0.8),
            _Detail("elevator", x=(0.60, 0.65), y=0.85, size=0.06, prob=0.8),
        ),
    ),
    "amphib_assault": _ShipClass(
        hull="carrier",
        lb=(6.0, 8.0),
        bow=(0.25, 0.45),
        stern_hw=(0.28, 0.38),
        shade=(125, 155),
        structs=(
            _Struct(
                x0=(0.30, 0.40),
                x1=(0.50, 0.58),
                w=(0.12, 0.22),
                y_off=0.28,
                shade_off=35,
            ),
        ),
        details=(
            _Detail("circle_spot", x=(0.20, 0.25), size=0.06, prob=0.9),
            _Detail("circle_spot", x=(0.40, 0.45), size=0.06, prob=0.9),
            _Detail("circle_spot", x=(0.60, 0.65), size=0.06, prob=0.8),
            _Detail("circle_spot", x=(0.75, 0.80), size=0.06, prob=0.7),
            _Detail("door", x=(0.95, 0.98), size=0.04, prob=0.6),
        ),
    ),
    # --- Military: landing / transport ---
    "lst_lpd": _ShipClass(
        hull="box",
        lb=(5.0, 7.0),
        bow=(0.15, 0.35),
        stern_hw=(0.25, 0.40),
        shade=(130, 165),
        structs=(
            _Struct(
                x0=(0.55, 0.65), x1=(0.72, 0.80), w=(0.35, 0.55), shade_off=30,
            ),
        ),
        details=(
            _Detail("door", x=(0.02, 0.05), size=0.03),
            _Detail("door", x=(0.95, 0.98), size=0.04, prob=0.7),
            _Detail("helipad", x=(0.82, 0.92), size=0.08, prob=0.8),
            _Detail("crane", x=(0.45, 0.52), y=0.75, size=0.03, prob=0.5),
        ),
    ),
    "supply": _ShipClass(
        hull="box",
        lb=(6.0, 8.0),
        bow=(0.2, 0.4),
        stern_hw=(0.15, 0.25),
        shade=(135, 170),
        structs=(
            _Struct(
                x0=(0.50, 0.58), x1=(0.68, 0.76), w=(0.30, 0.50), shade_off=25,
            ),
        ),
        details=(
            _Detail("crane", x=(0.25, 0.32), y=0.65, size=0.04, prob=0.8),
            _Detail("crane", x=(0.35, 0.42), y=0.35, size=0.04, prob=0.7),
            _Detail("crane", x=(0.78, 0.85), y=0.65, size=0.04, prob=0.5),
            _Detail("helipad", x=(0.85, 0.93), size=0.07, prob=0.6),
        ),
    ),
    # --- Fishing vessels ---
    "fishing_squid_jigger": _ShipClass(
        hull="fishing",
        lb=(5.0, 7.0),
        bow=(0.5, 0.8),
        stern_hw=(0.05, 0.15),
        shade=(130, 165),
        structs=(
            _Struct(x0=(0.25, 0.32), x1=(0.42, 0.50), w=(0.30, 0.50), shade_off=25),
        ),
        details=(
            # Lamp rows - port side
            _Detail("lamp", x=(0.15, 0.20), y=0.15, size=0.015, prob=0.9),
            _Detail("lamp", x=(0.25, 0.30), y=0.15, size=0.015, prob=0.9),
            _Detail("lamp", x=(0.35, 0.40), y=0.15, size=0.015, prob=0.9),
            _Detail("lamp", x=(0.45, 0.50), y=0.15, size=0.015, prob=0.9),
            _Detail("lamp", x=(0.55, 0.60), y=0.15, size=0.015, prob=0.8),
            _Detail("lamp", x=(0.65, 0.70), y=0.15, size=0.015, prob=0.7),
            # Lamp rows - starboard side
            _Detail("lamp", x=(0.15, 0.20), y=0.85, size=0.015, prob=0.9),
            _Detail("lamp", x=(0.25, 0.30), y=0.85, size=0.015, prob=0.9),
            _Detail("lamp", x=(0.35, 0.40), y=0.85, size=0.015, prob=0.9),
            _Detail("lamp", x=(0.45, 0.50), y=0.85, size=0.015, prob=0.9),
            _Detail("lamp", x=(0.55, 0.60), y=0.85, size=0.015, prob=0.8),
            _Detail("lamp", x=(0.65, 0.70), y=0.85, size=0.015, prob=0.7),
        ),
    ),
    "fishing_trawler": _ShipClass(
        hull="fishing_wide",
        lb=(4.5, 6.0),
        bow=(0.4, 0.7),
        stern_hw=(0.10, 0.20),
        shade=(130, 170),
        structs=(
            _Struct(x0=(0.18, 0.25), x1=(0.38, 0.45), w=(0.35, 0.55), shade_off=25),
        ),
        details=(
            _Detail("crane", x=(0.68, 0.78), y=0.35, size=0.05, prob=0.8),
            _Detail("crane", x=(0.68, 0.78), y=0.65, size=0.05, prob=0.6),
            _Detail("mast", x=(0.40, 0.48), size=0.035, prob=0.7),
        ),
    ),
    "fishing_purse_seiner": _ShipClass(
        hull="fishing_wide",
        lb=(5.0, 7.0),
        bow=(0.4, 0.65),
        stern_hw=(0.10, 0.20),
        shade=(130, 170),
        structs=(
            _Struct(x0=(0.20, 0.28), x1=(0.38, 0.46), w=(0.30, 0.50), shade_off=25),
            _Struct(
                x0=(0.50, 0.58),
                x1=(0.62, 0.68),
                w=(0.20, 0.35),
                y_off=0.15,
                shade_off=15,
                prob=0.6,
            ),
        ),
        details=(
            _Detail("crane", x=(0.60, 0.70), y=0.70, size=0.04, prob=0.7),
            _Detail("mast", x=(0.42, 0.50), size=0.03, prob=0.7),
        ),
    ),
    "fishing_longliner": _ShipClass(
        hull="fishing",
        lb=(5.0, 7.0),
        bow=(0.5, 0.75),
        stern_hw=(0.05, 0.15),
        shade=(130, 170),
        structs=(
            _Struct(x0=(0.20, 0.28), x1=(0.38, 0.46), w=(0.30, 0.50), shade_off=25),
        ),
        details=(
            _Detail("mast", x=(0.40, 0.48), size=0.03, prob=0.6),
            _Detail("crane", x=(0.70, 0.80), y=0.60, size=0.03, prob=0.5),
        ),
    ),
}


# ── Pure computation ─────────────────────────────────────────────────────


def _interpolate_hull(
    profile_key: str,
    bow_sharpness: float,
    stern_hw: float,
    n_points: int,
) -> NDArray[np.float64]:
    """Interpolate hull profile to *n_points* half-width values.

    Parameters
    ----------
    profile_key
        Key into ``_PROFILES``.
    bow_sharpness
        0 = blunt bow, 1 = sharp bow.  Modulates width near the bow.
    stern_hw
        Half-width at the stern tip (replaces last control point).
    n_points
        Number of rows to interpolate (= ship length in pixels).

    Returns
    -------
    NDArray[np.float64]
        Half-width at each row, values in [0, 0.5].
    """
    pts = list(_PROFILES[profile_key])
    # Sharper bow → narrower at early positions
    for i, (pos, hw) in enumerate(pts):
        if 0 < pos <= 0.15:
            factor = 0.5 + 0.5 * (1.0 - bow_sharpness)
            pts[i] = (pos, hw * factor)
    # Apply stern width
    pts[-1] = (pts[-1][0], stern_hw)

    xs = np.array([p[0] for p in pts])
    hws = np.array([p[1] for p in pts])
    t = np.linspace(0.0, 1.0, n_points)
    return np.interp(t, xs, hws)


def _build_hull_polygon(
    length_px: int,
    beam_px: int,
    half_widths: NDArray[np.float64],
    rng: random.Random,
    noise_scale: float = 0.005,
) -> list[tuple[int, int]]:
    """Convert half-width array to polygon vertices.

    Returns vertices traced clockwise: right (starboard) side bow→stern,
    then left (port) side stern→bow.
    """
    n = len(half_widths)
    noise = np.array([rng.gauss(0, noise_scale) for _ in range(n)])
    hw = np.clip(half_widths + noise, 0.0, 0.5)

    cx = beam_px / 2.0
    right: list[tuple[int, int]] = []
    left: list[tuple[int, int]] = []
    for i, w in enumerate(hw):
        y = int(i * (length_px - 1) / max(n - 1, 1))
        x_r = int(cx + w * beam_px)
        x_l = int(cx - w * beam_px)
        right.append((x_r, y))
        left.append((x_l, y))
    return right + list(reversed(left))


# ── Rendering helpers ────────────────────────────────────────────────────


def _draw_struct(
    draw: ImageDraw.ImageDraw,
    spec: _Struct,
    length_px: int,
    beam_px: int,
    base_shade: int,
    half_widths: NDArray[np.float64],
    rng: random.Random,
) -> None:
    """Draw one superstructure block, clipped to hull width."""
    x0 = rng.uniform(*spec.x0)
    x1 = rng.uniform(*spec.x1)
    if x0 >= x1:
        return
    w_frac = rng.uniform(*spec.w)

    y0 = int(x0 * length_px)
    y1 = int(x1 * length_px)

    # Hull half-width at block midpoint (for clipping)
    mid_idx = int((x0 + x1) / 2 * (len(half_widths) - 1))
    mid_idx = min(mid_idx, len(half_widths) - 1)
    hull_hw = half_widths[mid_idx]

    block_hw = w_frac * 0.5
    cx = 0.5 + spec.y_off
    edge_l = cx - block_hw
    edge_r = cx + block_hw

    # Clip to hull bounds with small margin
    edge_l = max(edge_l, 0.5 - hull_hw + 0.02)
    edge_r = min(edge_r, 0.5 + hull_hw - 0.02)
    if edge_l >= edge_r:
        return

    xl = int(edge_l * beam_px)
    xr = int(edge_r * beam_px)
    shade = min(255, base_shade + spec.shade_off + rng.randint(-5, 5))
    draw.rectangle([(xl, y0), (xr, y1)], fill=(shade, shade, shade, 255))


def _draw_detail(
    draw: ImageDraw.ImageDraw,
    detail: _Detail,
    length_px: int,
    beam_px: int,
    rng: random.Random,
) -> None:
    """Draw one detail element (mast, gun, helipad, …)."""
    x_pos = rng.uniform(*detail.x)
    y_row = int(x_pos * length_px)
    x_centre = int(detail.y * beam_px)
    sz = max(1, int(detail.size * length_px))
    ship_cx = beam_px // 2

    kind = detail.kind

    if kind == "mast":
        w = max(1, sz // 3)
        draw.rectangle(
            [(x_centre - w // 2, y_row - sz // 2),
             (x_centre + w // 2, y_row + sz // 2)],
            fill=(90, 90, 90, 255),
        )

    elif kind == "gun":
        r = max(1, sz // 2)
        draw.ellipse(
            [(x_centre - r, y_row - r), (x_centre + r, y_row + r)],
            fill=(100, 100, 100, 255),
        )

    elif kind in ("helipad", "circle_spot"):
        r = max(1, sz // 2)
        draw.ellipse(
            [(x_centre - r, y_row - r), (x_centre + r, y_row + r)],
            outline=(180, 180, 180, 255),
        )

    elif kind == "vls":
        rows_n = max(2, sz // 2)
        cols = 2
        cell = max(1, sz // rows_n)
        for ri in range(rows_n):
            for ci in range(cols):
                cx = x_centre - cols * cell // 2 + ci * cell
                cy = y_row - rows_n * cell // 2 + ri * cell
                draw.rectangle(
                    [(cx, cy), (cx + cell - 1, cy + cell - 1)],
                    fill=(140, 140, 140, 255),
                    outline=(110, 110, 110, 255),
                )

    elif kind == "crane":
        arm_w = max(1, sz // 4)
        # Vertical post
        draw.line(
            [(x_centre, y_row - sz), (x_centre, y_row + sz)],
            fill=(100, 100, 100, 255),
            width=arm_w,
        )
        # Horizontal boom
        draw.line(
            [(x_centre - sz // 2, y_row - sz),
             (x_centre + sz // 2, y_row - sz)],
            fill=(100, 100, 100, 255),
            width=arm_w,
        )

    elif kind == "lamp":
        r = max(1, sz)
        draw.ellipse(
            [(x_centre - r, y_row - r), (x_centre + r, y_row + r)],
            fill=(200, 200, 200, 255),
        )

    elif kind == "line":
        # Angled deck line (carriers)
        line_len = int(detail.size * length_px)
        x2 = x_centre + int(0.35 * beam_px)
        y2 = y_row + line_len
        draw.line(
            [(x_centre, y_row), (x2, y2)],
            fill=(170, 170, 170, 255),
            width=max(1, beam_px // 25),
        )

    elif kind == "door":
        w = int(0.6 * beam_px)
        draw.line(
            [(ship_cx - w // 2, y_row), (ship_cx + w // 2, y_row)],
            fill=(100, 100, 100, 255),
            width=max(1, length_px // 50),
        )

    elif kind == "elevator":
        el_sz = max(2, sz)
        draw.rectangle(
            [(x_centre - el_sz, y_row - el_sz // 2),
             (x_centre + el_sz, y_row + el_sz // 2)],
            outline=(155, 155, 155, 255),
        )


# ── Public API ───────────────────────────────────────────────────────────


def get_ship_classes() -> list[str]:
    """Return sorted list of available ship class names."""
    return sorted(SHIP_CLASSES)


def generate_ship_image(
    ship_class: str,
    length_px: int,
    rng: random.Random | None = None,
    hull_noise: float = 0.005,
) -> Image.Image:
    """Generate a single ship as a transparent RGBA image.

    The image is sized so that ``width = beam_px`` and ``height = length_px``,
    i.e. the image *is* the bounding box.  Bow is at the top (y = 0).

    Parameters
    ----------
    ship_class
        Key in :data:`SHIP_CLASSES`.
    length_px
        Ship length in pixels (= image height).
    rng
        Random state.  Created from system entropy when *None*.
    hull_noise
        Standard deviation of hull outline perturbation.

    Returns
    -------
    PIL.Image.Image
        RGBA image with transparent background.
    """
    if rng is None:
        rng = random.Random()

    cls = SHIP_CLASSES[ship_class]

    # Sample per-instance parameters
    lb_ratio = rng.uniform(*cls.lb)
    beam_px = max(4, int(length_px / lb_ratio))
    bow_sharpness = rng.uniform(*cls.bow)
    stern_hw = rng.uniform(*cls.stern_hw)
    base_shade = rng.randint(*cls.shade)

    # Hull outline
    half_widths = _interpolate_hull(cls.hull, bow_sharpness, stern_hw, length_px)
    polygon = _build_hull_polygon(length_px, beam_px, half_widths, rng, hull_noise)

    # Canvas
    img = Image.new("RGBA", (beam_px, length_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1) Hull
    draw.polygon(polygon, fill=(base_shade, base_shade, base_shade, 255))

    # 2) Superstructures
    for s in cls.structs:
        if rng.random() < s.prob:
            _draw_struct(draw, s, length_px, beam_px, base_shade, half_widths, rng)

    # 3) Details
    for d in cls.details:
        if rng.random() < d.prob:
            _draw_detail(draw, d, length_px, beam_px, rng)

    # 4) Light pixel‐level noise over opaque area for natural look
    arr = np.array(img)
    opaque = arr[:, :, 3] > 0
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    noise_vals = np_rng.integers(-3, 4, size=arr.shape[:2], dtype=np.int16)
    for c in range(3):
        ch = arr[:, :, c].astype(np.int16)
        ch[opaque] += noise_vals[opaque]
        arr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


def generate_ships(
    output_dir: Path,
    count: int,
    image_size: tuple[int, int],
    types: dict[str, float] | None = None,
    seed: int | None = None,
    hull_noise: float = 0.005,
) -> None:
    """Generate synthetic ship images and save as transparent PNGs.

    Parameters
    ----------
    output_dir
        Destination directory (created if absent).
    count
        Number of images to generate.
    image_size
        ``(min_length_px, max_length_px)`` range for ship length.
    types
        ``{ship_class: weight}`` mapping.  Equal weights for all classes
        when *None*.
    seed
        Random seed for reproducibility.
    hull_noise
        Standard deviation of hull outline perturbation.

    Raises
    ------
    ValueError
        If *types* contains an unknown ship class name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if types is None:
        classes = get_ship_classes()
        weights = [1.0] * len(classes)
    else:
        for name in types:
            if name not in SHIP_CLASSES:
                msg = f"Unknown ship class: {name!r}. Available: {get_ship_classes()}"
                raise ValueError(msg)
        classes = list(types)
        weights = [types[c] for c in classes]

    min_len, max_len = image_size
    counters: dict[str, int] = {}

    for i in range(count):
        ship_class = rng.choices(classes, weights=weights, k=1)[0]
        length_px = rng.randint(min_len, max_len)
        img = generate_ship_image(ship_class, length_px, rng=rng, hull_noise=hull_noise)

        counters[ship_class] = counters.get(ship_class, 0) + 1
        filename = f"{ship_class}_{counters[ship_class]:05d}.png"
        img.save(output_dir / filename)

        if (i + 1) % 100 == 0 or (i + 1) == count:
            logger.info("Generated %d / %d images", i + 1, count)

    logger.info("Ship counts: %s", dict(sorted(counters.items())))
