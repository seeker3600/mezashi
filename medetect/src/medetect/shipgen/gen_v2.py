"""Synthetic ship silhouette generator using shape grammar.

Generates ship top-down silhouettes as SVG (default) or transparent RGBA PNG.
Ships are composed from geometric primitives: hull polygon + superstructure
rectangles + detail marks.

Ship classes and their colour palettes are derived from the design documents
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

# ── Type aliases ─────────────────────────────────────────────────────────

Color = tuple[int, int, int]

# ── Hull profile control points ──────────────────────────────────────────
# Each profile: list of (position_from_bow ∈ [0,1], half_width ∈ [0,0.5]).

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

# ── Drawing primitives ───────────────────────────────────────────────────


@dataclass(frozen=True)
class _Polygon:
    """Filled polygon."""

    points: tuple[tuple[float, float], ...]
    fill: Color


@dataclass(frozen=True)
class _Rect:
    """Rectangle with optional fill and stroke."""

    x: float
    y: float
    w: float
    h: float
    fill: Color | None = None
    stroke: Color | None = None
    stroke_w: float = 1.0


@dataclass(frozen=True)
class _Ellipse:
    """Ellipse with optional fill and stroke."""

    cx: float
    cy: float
    rx: float
    ry: float
    fill: Color | None = None
    stroke: Color | None = None
    stroke_w: float = 1.0


@dataclass(frozen=True)
class _Line:
    """Line segment."""

    x1: float
    y1: float
    x2: float
    y2: float
    stroke: Color
    width: float = 1.0


_Prim = _Polygon | _Rect | _Ellipse | _Line

# ── Ship class template data ─────────────────────────────────────────────


@dataclass(frozen=True)
class _Struct:
    """Superstructure block placement rule."""

    x0: tuple[float, float]  # start position range along length
    x1: tuple[float, float]  # end position range along length
    w: tuple[float, float]  # width as fraction of beam
    y_off: float = 0.0  # lateral offset from centre (-0.5..0.5)
    shade_off: int = 30  # brightness offset from hull (used when color is None)
    color: Color | None = None  # absolute colour override
    prob: float = 1.0  # probability of placement


@dataclass(frozen=True)
class _Detail:
    """Small detail element placement rule."""

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
    colors: tuple[Color, ...]  # candidate hull base RGB colours
    color_var: int  # per-channel random ±variation
    structs: tuple[_Struct, ...]
    details: tuple[_Detail, ...]


# ── Colour palettes ──────────────────────────────────────────────────────

_MILITARY_GRAY: tuple[Color, ...] = (
    (139, 141, 142),  # Haze Gray (USN standard)
    (150, 155, 158),  # Light gray
    (120, 125, 130),  # Medium gray
    (132, 136, 140),  # Cool gray
    (145, 148, 152),  # Neutral gray
)

_DARK_MILITARY: tuple[Color, ...] = (
    (115, 118, 122),
    (108, 112, 118),
    (125, 128, 132),
)

_CARRIER_DECK: tuple[Color, ...] = (
    (95, 98, 102),
    (105, 108, 112),
    (88, 91, 95),
)

_AMPHIB_GRAY: tuple[Color, ...] = (
    (110, 113, 118),
    (120, 123, 128),
    (100, 104, 108),
)

_WARM_GRAY: tuple[Color, ...] = (
    (135, 132, 128),
    (145, 142, 138),
    (125, 122, 118),
)

_SUPPLY_GRAY: tuple[Color, ...] = (
    (140, 143, 138),
    (150, 150, 145),
    (130, 135, 130),
)

_FISHING_BLUE: tuple[Color, ...] = (
    (35, 50, 90),
    (45, 60, 105),
    (25, 40, 75),
    (55, 70, 115),
)

_FISHING_VARIED: tuple[Color, ...] = (
    (40, 60, 110),
    (50, 75, 130),
    (130, 42, 35),
    (150, 55, 40),
    (175, 178, 170),
    (45, 75, 58),
)

_FISHING_BLUE_DARK: tuple[Color, ...] = (
    (35, 55, 100),
    (50, 70, 120),
    (45, 48, 55),
    (60, 80, 130),
)

_FISHING_MIXED: tuple[Color, ...] = (
    (40, 60, 110),
    (145, 48, 38),
    (170, 172, 165),
    (55, 75, 120),
)

_FISHING_CABIN: Color = (195, 198, 190)

# ── Ship class registry ──────────────────────────────────────────────────

SHIP_CLASSES: dict[str, _ShipClass] = {
    # --- Military: normal warships ---
    "patrol": _ShipClass(
        hull="warship",
        lb=(6.0, 9.0),
        bow=(0.6, 0.9),
        stern_hw=(0.05, 0.15),
        colors=_MILITARY_GRAY,
        color_var=8,
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
        colors=_MILITARY_GRAY,
        color_var=8,
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
        colors=_MILITARY_GRAY,
        color_var=8,
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
        colors=_DARK_MILITARY,
        color_var=8,
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
        colors=_CARRIER_DECK,
        color_var=6,
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
        colors=_AMPHIB_GRAY,
        color_var=7,
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
        colors=_WARM_GRAY,
        color_var=8,
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
        colors=_SUPPLY_GRAY,
        color_var=10,
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
        colors=_FISHING_BLUE,
        color_var=12,
        structs=(
            _Struct(
                x0=(0.25, 0.32), x1=(0.42, 0.50), w=(0.30, 0.50),
                color=_FISHING_CABIN,
            ),
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
        colors=_FISHING_VARIED,
        color_var=15,
        structs=(
            _Struct(
                x0=(0.18, 0.25), x1=(0.38, 0.45), w=(0.35, 0.55),
                color=_FISHING_CABIN,
            ),
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
        colors=_FISHING_BLUE_DARK,
        color_var=12,
        structs=(
            _Struct(
                x0=(0.20, 0.28), x1=(0.38, 0.46), w=(0.30, 0.50),
                color=_FISHING_CABIN,
            ),
            _Struct(
                x0=(0.50, 0.58),
                x1=(0.62, 0.68),
                w=(0.20, 0.35),
                y_off=0.15,
                color=_FISHING_CABIN,
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
        colors=_FISHING_MIXED,
        color_var=15,
        structs=(
            _Struct(
                x0=(0.20, 0.28), x1=(0.38, 0.46), w=(0.30, 0.50),
                color=_FISHING_CABIN,
            ),
        ),
        details=(
            _Detail("mast", x=(0.40, 0.48), size=0.03, prob=0.6),
            _Detail("crane", x=(0.70, 0.80), y=0.60, size=0.03, prob=0.5),
        ),
    ),
}


# ── Colour helpers ───────────────────────────────────────────────────────


def _pick_color(
    candidates: tuple[Color, ...],
    var: int,
    rng: random.Random,
) -> Color:
    """Choose a base colour and apply per-channel random variation."""
    base = candidates[rng.randrange(len(candidates))]
    return (
        max(0, min(255, base[0] + rng.randint(-var, var))),
        max(0, min(255, base[1] + rng.randint(-var, var))),
        max(0, min(255, base[2] + rng.randint(-var, var))),
    )


def _offset_color(base: Color, offset: int) -> Color:
    """Shift all channels by *offset*, clamping to [0, 255]."""
    return (
        max(0, min(255, base[0] + offset)),
        max(0, min(255, base[1] + offset)),
        max(0, min(255, base[2] + offset)),
    )


def _vary_color(base: Color, var: int, rng: random.Random) -> Color:
    """Apply small random variation to a fixed colour."""
    return (
        max(0, min(255, base[0] + rng.randint(-var, var))),
        max(0, min(255, base[1] + rng.randint(-var, var))),
        max(0, min(255, base[2] + rng.randint(-var, var))),
    )


def _rgb(c: Color) -> str:
    """Format colour as SVG ``rgb(r,g,b)`` string."""
    return f"rgb({c[0]},{c[1]},{c[2]})"


# ── Hull computation ─────────────────────────────────────────────────────


def _interpolate_hull(
    profile_key: str,
    bow_sharpness: float,
    stern_hw: float,
    n_points: int,
) -> NDArray[np.float64]:
    """Interpolate hull profile to *n_points* half-width values.

    Returns
    -------
    NDArray[np.float64]
        Half-width at each row, values in [0, 0.5].
    """
    pts = list(_PROFILES[profile_key])
    for i, (pos, hw) in enumerate(pts):
        if 0 < pos <= 0.15:
            factor = 0.5 + 0.5 * (1.0 - bow_sharpness)
            pts[i] = (pos, hw * factor)
    pts[-1] = (pts[-1][0], stern_hw)

    xs = np.array([p[0] for p in pts])
    hws = np.array([p[1] for p in pts])
    t = np.linspace(0.0, 1.0, n_points)
    return np.interp(t, xs, hws)


def _build_hull_polygon(
    length: float,
    beam: float,
    half_widths: NDArray[np.float64],
    rng: random.Random,
    noise_scale: float = 0.005,
) -> list[tuple[float, float]]:
    """Convert half-width array to polygon vertices (float coordinates).

    Returns vertices traced clockwise: right (starboard) side bow→stern,
    then left (port) side stern→bow.
    """
    n = len(half_widths)
    noise = np.array([rng.gauss(0, noise_scale) for _ in range(n)])
    hw = np.clip(half_widths + noise, 0.0, 0.5)

    cx = beam / 2.0
    right: list[tuple[float, float]] = []
    left: list[tuple[float, float]] = []
    for i, w in enumerate(hw):
        y = i * (length - 1) / max(n - 1, 1)
        x_r = cx + w * beam
        x_l = cx - w * beam
        right.append((x_r, y))
        left.append((x_l, y))
    return right + list(reversed(left))


# ── Primitive generation ─────────────────────────────────────────────────


def _make_struct_prim(
    spec: _Struct,
    length: float,
    beam: float,
    hull_color: Color,
    half_widths: NDArray[np.float64],
    rng: random.Random,
) -> _Rect | None:
    """Resolve a superstructure rule to a _Rect primitive or *None*."""
    x0 = rng.uniform(*spec.x0)
    x1 = rng.uniform(*spec.x1)
    if x0 >= x1:
        return None
    w_frac = rng.uniform(*spec.w)

    y0 = x0 * length
    y1 = x1 * length

    mid_idx = int((x0 + x1) / 2 * (len(half_widths) - 1))
    mid_idx = min(mid_idx, len(half_widths) - 1)
    hull_hw = half_widths[mid_idx]

    block_hw = w_frac * 0.5
    centre = 0.5 + spec.y_off
    edge_l = max(centre - block_hw, 0.5 - hull_hw + 0.02)
    edge_r = min(centre + block_hw, 0.5 + hull_hw - 0.02)
    if edge_l >= edge_r:
        return None

    rx = edge_l * beam
    rw = (edge_r - edge_l) * beam
    rh = y1 - y0

    if spec.color is not None:
        c = _vary_color(spec.color, 5, rng)
    else:
        c = _offset_color(hull_color, spec.shade_off + rng.randint(-5, 5))

    return _Rect(x=rx, y=y0, w=rw, h=rh, fill=c)


def _make_detail_prims(
    detail: _Detail,
    length: float,
    beam: float,
    hull_color: Color,
    rng: random.Random,
) -> list[_Prim]:
    """Resolve a detail rule to drawing primitives."""
    x_pos = rng.uniform(*detail.x)
    y_row = x_pos * length
    x_centre = detail.y * beam
    sz = max(1.0, detail.size * length)
    ship_cx = beam / 2.0

    kind = detail.kind
    prims: list[_Prim] = []

    if kind == "mast":
        w = max(1.0, sz / 3)
        prims.append(_Rect(
            x=x_centre - w / 2, y=y_row - sz / 2, w=w, h=sz,
            fill=_offset_color(hull_color, -40),
        ))

    elif kind == "gun":
        r = max(1.0, sz / 2)
        prims.append(_Ellipse(
            cx=x_centre, cy=y_row, rx=r, ry=r,
            fill=_offset_color(hull_color, -25),
        ))

    elif kind in ("helipad", "circle_spot"):
        r = max(1.0, sz / 2)
        prims.append(_Ellipse(
            cx=x_centre, cy=y_row, rx=r, ry=r,
            stroke=_offset_color(hull_color, 35),
        ))

    elif kind == "vls":
        rows_n = max(2, int(sz / 2))
        cols = 2
        cell = max(1.0, sz / rows_n)
        for ri in range(rows_n):
            for ci in range(cols):
                cx = x_centre - cols * cell / 2 + ci * cell
                cy = y_row - rows_n * cell / 2 + ri * cell
                prims.append(_Rect(
                    x=cx, y=cy, w=cell, h=cell,
                    fill=_offset_color(hull_color, 10),
                    stroke=_offset_color(hull_color, -15),
                ))

    elif kind == "crane":
        arm_w = max(1.0, sz / 4)
        clr = _offset_color(hull_color, -30)
        prims.append(_Line(
            x1=x_centre, y1=y_row - sz, x2=x_centre, y2=y_row + sz,
            stroke=clr, width=arm_w,
        ))
        prims.append(_Line(
            x1=x_centre - sz / 2, y1=y_row - sz,
            x2=x_centre + sz / 2, y2=y_row - sz,
            stroke=clr, width=arm_w,
        ))

    elif kind == "lamp":
        r = max(1.0, sz)
        prims.append(_Ellipse(
            cx=x_centre, cy=y_row, rx=r, ry=r,
            fill=(210, 210, 200),
        ))

    elif kind == "line":
        line_len = detail.size * length
        x2 = x_centre + 0.35 * beam
        y2 = y_row + line_len
        prims.append(_Line(
            x1=x_centre, y1=y_row, x2=x2, y2=y2,
            stroke=_offset_color(hull_color, 25),
            width=max(1.0, beam / 25),
        ))

    elif kind == "door":
        w = 0.6 * beam
        prims.append(_Line(
            x1=ship_cx - w / 2, y1=y_row,
            x2=ship_cx + w / 2, y2=y_row,
            stroke=_offset_color(hull_color, -30),
            width=max(1.0, length / 50),
        ))

    elif kind == "elevator":
        el_sz = max(2.0, sz)
        prims.append(_Rect(
            x=x_centre - el_sz, y=y_row - el_sz / 2,
            w=2 * el_sz, h=el_sz,
            stroke=_offset_color(hull_color, 15),
        ))

    return prims


def _generate_ship_prims(
    ship_class: str,
    length: float,
    rng: random.Random,
    hull_noise: float,
) -> tuple[float, float, Color, list[_Prim]]:
    """Build geometry primitives for one ship.

    Returns ``(beam, length, hull_color, primitives)``.
    """
    cls = SHIP_CLASSES[ship_class]

    lb_ratio = rng.uniform(*cls.lb)
    beam = max(4.0, length / lb_ratio)
    bow_sharpness = rng.uniform(*cls.bow)
    stern_hw = rng.uniform(*cls.stern_hw)
    hull_color = _pick_color(cls.colors, cls.color_var, rng)

    n_pts = max(8, round(length))
    half_widths = _interpolate_hull(cls.hull, bow_sharpness, stern_hw, n_pts)
    hull_pts = _build_hull_polygon(length, beam, half_widths, rng, hull_noise)

    prims: list[_Prim] = [_Polygon(points=tuple(hull_pts), fill=hull_color)]

    for s in cls.structs:
        if rng.random() < s.prob:
            prim = _make_struct_prim(s, length, beam, hull_color, half_widths, rng)
            if prim is not None:
                prims.append(prim)

    for d in cls.details:
        if rng.random() < d.prob:
            prims.extend(_make_detail_prims(d, length, beam, hull_color, rng))

    return beam, length, hull_color, prims


# ── Renderers ────────────────────────────────────────────────────────────


def _render_svg(beam: float, length: float, prims: list[_Prim]) -> str:
    """Render primitives to an SVG string.

    The hull polygon is used as a clip path so that superstructures and
    details never extend beyond the ship outline.
    """
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' viewBox="0 0 {beam:.2f} {length:.2f}">',
    ]

    # Extract hull polygon for clip path (always first primitive)
    hull = prims[0]
    assert isinstance(hull, _Polygon)
    hull_pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in hull.points)

    parts.append("  <defs>")
    parts.append(f'    <clipPath id="hull">')
    parts.append(f'      <polygon points="{hull_pts_str}"/>')
    parts.append("    </clipPath>")
    parts.append("  </defs>")

    # Hull fill
    parts.append(
        f'  <polygon points="{hull_pts_str}" fill="{_rgb(hull.fill)}"/>'
    )

    # Clipped group for all upper elements
    parts.append('  <g clip-path="url(#hull)">')
    for p in prims[1:]:
        if isinstance(p, _Polygon):
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in p.points)
            parts.append(f'    <polygon points="{pts}" fill="{_rgb(p.fill)}"/>')
        elif isinstance(p, _Rect):
            fill = f'fill="{_rgb(p.fill)}"' if p.fill else 'fill="none"'
            stroke = (
                f' stroke="{_rgb(p.stroke)}" stroke-width="{p.stroke_w:.1f}"'
                if p.stroke else ""
            )
            parts.append(
                f'    <rect x="{p.x:.2f}" y="{p.y:.2f}"'
                f' width="{p.w:.2f}" height="{p.h:.2f}"'
                f" {fill}{stroke}/>"
            )
        elif isinstance(p, _Ellipse):
            fill = f'fill="{_rgb(p.fill)}"' if p.fill else 'fill="none"'
            stroke = (
                f' stroke="{_rgb(p.stroke)}" stroke-width="{p.stroke_w:.1f}"'
                if p.stroke else ""
            )
            parts.append(
                f'    <ellipse cx="{p.cx:.2f}" cy="{p.cy:.2f}"'
                f' rx="{p.rx:.2f}" ry="{p.ry:.2f}"'
                f" {fill}{stroke}/>"
            )
        elif isinstance(p, _Line):
            parts.append(
                f'    <line x1="{p.x1:.2f}" y1="{p.y1:.2f}"'
                f' x2="{p.x2:.2f}" y2="{p.y2:.2f}"'
                f' stroke="{_rgb(p.stroke)}" stroke-width="{p.width:.1f}"/>'
            )
    parts.append("  </g>")
    parts.append("</svg>")
    return "\n".join(parts)


def _render_pil(
    beam: float,
    length: float,
    prims: list[_Prim],
    rng: random.Random,
) -> Image.Image:
    """Render primitives to a transparent RGBA PIL Image with pixel noise."""
    w, h = max(1, round(beam)), max(1, round(length))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for p in prims:
        if isinstance(p, _Polygon):
            draw.polygon(
                [(x, y) for x, y in p.points],
                fill=(*p.fill, 255),
            )
        elif isinstance(p, _Rect):
            box = [(p.x, p.y), (p.x + p.w, p.y + p.h)]
            if p.fill:
                draw.rectangle(box, fill=(*p.fill, 255))
            if p.stroke:
                draw.rectangle(box, outline=(*p.stroke, 255))
        elif isinstance(p, _Ellipse):
            box = [(p.cx - p.rx, p.cy - p.ry), (p.cx + p.rx, p.cy + p.ry)]
            if p.fill:
                draw.ellipse(box, fill=(*p.fill, 255))
            if p.stroke:
                draw.ellipse(box, outline=(*p.stroke, 255))
        elif isinstance(p, _Line):
            draw.line(
                [(p.x1, p.y1), (p.x2, p.y2)],
                fill=(*p.stroke, 255),
                width=max(1, round(p.width)),
            )

    # Light pixel-level noise over opaque area
    arr = np.array(img)
    opaque = arr[:, :, 3] > 0
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    noise_vals = np_rng.integers(-3, 4, size=arr.shape[:2], dtype=np.int16)
    for c in range(3):
        ch = arr[:, :, c].astype(np.int16)
        ch[opaque] += noise_vals[opaque]
        arr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, "RGBA")


# ── Public API ───────────────────────────────────────────────────────────


def get_ship_classes() -> list[str]:
    """Return sorted list of available ship class names."""
    return sorted(SHIP_CLASSES)


def generate_ship_svg(
    ship_class: str,
    length: float = 100.0,
    rng: random.Random | None = None,
    hull_noise: float = 0.005,
) -> str:
    """Generate a single ship as an SVG string.

    The ``viewBox`` is set to the tight bounding box (beam × length)
    so the SVG can be scaled to any resolution.

    Parameters
    ----------
    ship_class
        Key in :data:`SHIP_CLASSES`.
    length
        Ship length in abstract SVG units (controls outline smoothness).
    rng
        Random state.  Created from system entropy when *None*.
    hull_noise
        Standard deviation of hull outline perturbation.

    Returns
    -------
    str
        Complete SVG document.
    """
    if rng is None:
        rng = random.Random()
    beam, length, _color, prims = _generate_ship_prims(
        ship_class, length, rng, hull_noise,
    )
    return _render_svg(beam, length, prims)


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
    beam, length, _color, prims = _generate_ship_prims(
        ship_class, float(length_px), rng, hull_noise,
    )
    return _render_pil(beam, length, prims, rng)


def generate_ships(
    output_dir: Path,
    count: int,
    image_size: tuple[int, int],
    types: dict[str, float] | None = None,
    seed: int | None = None,
    hull_noise: float = 0.005,
    fmt: str = "svg",
) -> None:
    """Generate synthetic ship images and save to *output_dir*.

    Parameters
    ----------
    output_dir
        Destination directory (created if absent).
    count
        Number of images to generate.
    image_size
        ``(min_length, max_length)`` range for ship length.
    types
        ``{ship_class: weight}`` mapping.  Equal weights for all classes
        when *None*.
    seed
        Random seed for reproducibility.
    hull_noise
        Standard deviation of hull outline perturbation.
    fmt
        Output format: ``"svg"`` (default) or ``"png"``.

    Raises
    ------
    ValueError
        If *types* contains an unknown ship class or *fmt* is invalid.
    """
    if fmt not in ("svg", "png"):
        msg = f"Unsupported format: {fmt!r}. Use 'svg' or 'png'."
        raise ValueError(msg)

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
        length = rng.randint(min_len, max_len)

        counters[ship_class] = counters.get(ship_class, 0) + 1

        if fmt == "svg":
            svg = generate_ship_svg(
                ship_class, float(length), rng=rng, hull_noise=hull_noise,
            )
            filename = f"{ship_class}_{counters[ship_class]:05d}.svg"
            (output_dir / filename).write_text(svg, encoding="utf-8")
        else:
            img = generate_ship_image(
                ship_class, length, rng=rng, hull_noise=hull_noise,
            )
            filename = f"{ship_class}_{counters[ship_class]:05d}.png"
            img.save(output_dir / filename)

        if (i + 1) % 100 == 0 or (i + 1) == count:
            logger.info("Generated %d / %d images", i + 1, count)

    logger.info("Ship counts: %s", dict(sorted(counters.items())))
