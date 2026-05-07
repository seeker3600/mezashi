"""Synthetic ship silhouette generator — SVG vector output.

Generates resolution-independent SVG images of ship top-down silhouettes
for object detection training.  Ships are composed from geometric
primitives: hull polygon + superstructure rects + detail marks, with
realistic colour palettes per ship family.

Coordinate system
-----------------
SVG viewBox = ``0 0 1 {lb_ratio}``.

* x ∈ [0, 1]  — beam direction (port = 0, starboard = 1)
* y ∈ [0, lb]  — length direction (bow = 0, stern = lb)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from numpy.typing import NDArray

from medetect.shipgen.hull import build_hull_points, interpolate_hull
from medetect.shipgen.ship_class import (
    SHIP_CLASSES,
    Detail,
    ShipColors,
    Struct,
    sample_colors,
    sample_ship_appearance_variant,
)


logger = logging.getLogger(__name__)

# Fraction of hull depth used to translate superstructure positions under
# off-nadir viewing.  beam_shift = side_component * _STRUCT_HEIGHT_FRAC
# Typical superstructure height / beam: destroyer ~0.5-0.7, merchant ~0.3-0.5.
# 0.45 is a representative mid-range value producing clearly visible shifts.
# マイナスにしないと船腹方向に移動しちゃうから手修正した。バグじゃないよ。
_STRUCT_HEIGHT_FRAC: float = -0.5

_DEBUG_RECT_COLORS: tuple[tuple[int, int, int], ...] = (
    (220, 48, 48),
    (48, 180, 72),
    (54, 104, 224),
    (236, 208, 48),
)

# ── SVG formatting helpers ───────────────────────────────────────────────


def _f(v: float) -> str:
    """Format a float for SVG coordinates (4 decimal places, strip trailing zeros)."""
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _polygon_attr(pts: list[tuple[float, float]]) -> str:
    return " ".join(f"{_f(x)},{_f(y)}" for x, y in pts)


def _rgb_css(color: tuple[int, int, int]) -> str:
    r, g, b = color
    return f"rgb({r},{g},{b})"


def _debug_rect_points(lb_ratio: float) -> list[tuple[float, float]]:
    return [(0.03, 0.0), (0.97, 0.0), (0.97, lb_ratio), (0.03, lb_ratio)]


def _inset_hull_points(
    hull_pts: list[tuple[float, float]],
    inset: float,
) -> list[tuple[float, float]]:
    """Move hull points inward toward the centerline by a fixed beam offset."""
    if inset <= 0.0:
        return list(hull_pts)

    inset_pts: list[tuple[float, float]] = []
    for x, y in hull_pts:
        offset = x - 0.5
        if abs(offset) < 1e-6:
            inset_pts.append((0.5, y))
            continue
        inner_offset = max(abs(offset) - inset, 0.0)
        inset_pts.append((0.5 + (1.0 if offset > 0.0 else -1.0) * inner_offset, y))
    return inset_pts


def _segment_index_bounds(point_count: int, t_start: float, t_end: float) -> tuple[int, int]:
    i0 = max(0, min(int(round(t_start * (point_count - 1))), point_count - 1))
    i1 = max(i0, min(int(round(t_end * (point_count - 1))), point_count - 1))
    return i0, i1


def _write_hull_edge_band(
    out: StringIO,
    hull_pts: list[tuple[float, float]],
    inner_pts: list[tuple[float, float]],
    *,
    side: str,
    fill: str,
    role: str,
    t_start: float = 0.0,
    t_end: float = 1.0,
    side_tag: str | None = None,
) -> None:
    """Write a narrow hull-edge band for one side of the ship."""
    if len(hull_pts) != len(inner_pts) or len(hull_pts) < 4:
        return

    point_count = len(hull_pts) // 2
    i0, i1 = _segment_index_bounds(point_count, t_start, t_end)
    if i1 - i0 < 1:
        return

    if side == "starboard":
        outer = [hull_pts[i] for i in range(i0, i1 + 1)]
        inner = [inner_pts[i] for i in range(i1, i0 - 1, -1)]
    elif side == "port":
        outer = [hull_pts[2 * point_count - 1 - i] for i in range(i0, i1 + 1)]
        inner = [inner_pts[2 * point_count - 1 - i] for i in range(i1, i0 - 1, -1)]
    else:
        msg = f"Unsupported hull edge side: {side!r}"
        raise ValueError(msg)

    attrs = f' data-side="{side_tag}"' if side_tag is not None else ""
    pts = outer + inner
    if len(pts) >= 3:
        out.write(
            f'  <polygon points="{_polygon_attr(pts)}" fill="{fill}" data-role="{role}"{attrs}/>\n'
        )


def _write_trim_cap(
    out: StringIO,
    hull_pts: list[tuple[float, float]],
    inner_pts: list[tuple[float, float]],
    *,
    index: int,
    fill: str,
    role: str,
) -> None:
    """Write a short cross-ship cap that closes a trim segment."""
    if len(hull_pts) != len(inner_pts) or len(hull_pts) < 4:
        return

    point_count = len(hull_pts) // 2
    idx = max(0, min(index, point_count - 1))
    pts = [
        hull_pts[idx],
        hull_pts[2 * point_count - 1 - idx],
        inner_pts[2 * point_count - 1 - idx],
        inner_pts[idx],
    ]
    unique = {(round(x, 6), round(y, 6)) for x, y in pts}
    if len(unique) < 3:
        return
    out.write(
        f'  <polygon points="{_polygon_attr(pts)}" fill="{fill}" data-role="{role}"/>\n'
    )


def _write_hull_trim(
    out: StringIO,
    hull_pts: list[tuple[float, float]],
    colors: ShipColors,
) -> None:
    """Write sampled hull trim and one-sided visible hull colour bands."""
    trim = colors.trim

    primary_fill = trim.primary_css()
    if primary_fill is not None and trim.primary_width > 0.0:
        primary_inner = _inset_hull_points(hull_pts, trim.primary_width)
        point_count = len(hull_pts) // 2
        if trim.primary_mode == "perimeter":
            _write_hull_edge_band(
                out,
                hull_pts,
                primary_inner,
                side="starboard",
                fill=primary_fill,
                role="hull-trim",
            )
            _write_hull_edge_band(
                out,
                hull_pts,
                primary_inner,
                side="port",
                fill=primary_fill,
                role="hull-trim",
            )
            _write_trim_cap(out, hull_pts, primary_inner, index=0, fill=primary_fill, role="hull-trim")
            _write_trim_cap(
                out,
                hull_pts,
                primary_inner,
                index=point_count - 1,
                fill=primary_fill,
                role="hull-trim",
            )
        elif trim.primary_mode == "bow":
            _write_hull_edge_band(
                out,
                hull_pts,
                primary_inner,
                side="starboard",
                fill=primary_fill,
                role="bow-trim",
                t_end=trim.bow_extent,
            )
            _write_hull_edge_band(
                out,
                hull_pts,
                primary_inner,
                side="port",
                fill=primary_fill,
                role="bow-trim",
                t_end=trim.bow_extent,
            )
            _write_trim_cap(out, hull_pts, primary_inner, index=0, fill=primary_fill, role="bow-trim")

    side_fill = trim.side_css()
    if side_fill is not None and trim.side_width > 0.0 and trim.visible_side != "none":
        side_inner = _inset_hull_points(hull_pts, trim.side_width)
        _write_hull_edge_band(
            out,
            hull_pts,
            side_inner,
            side=trim.visible_side,
            fill=side_fill,
            role="side-trim",
            t_start=trim.side_start,
            t_end=trim.side_end,
            side_tag=trim.visible_side,
        )


# ── SVG element writers ──────────────────────────────────────────────────


@dataclass(frozen=True)
class _ResolvedStructRect:
    el: float
    er: float
    y0: float
    y1: float
    brightness_off: int


def _apply_oversized_struct_geometry(
    x0: float,
    x1: float,
    w_frac: float,
) -> tuple[float, float, float]:
    span = x1 - x0
    if span <= 0.0:
        return x0, x1, w_frac

    center = (x0 + x1) * 0.5
    target_span = min(max(span * 1.9, 0.46), 0.60)
    x0 = max(center - target_span * 0.5, 0.10)
    x1 = min(center + target_span * 0.5, 0.92)
    if x1 - x0 < target_span:
        if x0 <= 0.10:
            x1 = min(x0 + target_span, 0.96)
        else:
            x0 = max(x1 - target_span, 0.04)
    w_frac = min(max(w_frac * 1.45, 0.78), 0.96)
    return x0, x1, w_frac


def _resolve_struct_rect(
    spec: Struct,
    lb_ratio: float,
    half_widths: NDArray,
    rng: random.Random,
    *,
    oversized_variant: bool = False,
    beam_shift: float = 0.0,
) -> _ResolvedStructRect | None:
    x0 = rng.uniform(*spec.x0)
    x1 = rng.uniform(*spec.x1)
    if x0 >= x1:
        return None
    w_frac = rng.uniform(*spec.w)
    if oversized_variant:
        x0, x1, w_frac = _apply_oversized_struct_geometry(x0, x1, w_frac)

    n = len(half_widths)
    mid_idx = min(int((x0 + x1) / 2 * (n - 1)), n - 1)
    hull_hw = float(half_widths[mid_idx])

    block_hw = w_frac * 0.5
    cx = 0.5 + spec.y_off + beam_shift
    # Near-side clamp is intentionally removed when beam_shift is non-zero:
    # off-nadir geometry should allow the near (visible) side of a structure
    # to protrude beyond the hull silhouette, which is physically correct.
    if beam_shift > 0.0:
        # Sensor sees starboard side — starboard (right) edge is free to overhang.
        el = max(cx - block_hw, 0.5 - hull_hw + 0.02)
        er = cx + block_hw
    elif beam_shift < 0.0:
        # Sensor sees port side — port (left) edge is free to overhang.
        el = cx - block_hw
        er = min(cx + block_hw, 0.5 + hull_hw - 0.02)
    else:
        el = max(cx - block_hw, 0.5 - hull_hw + 0.02)
        er = min(cx + block_hw, 0.5 + hull_hw - 0.02)
    if el >= er:
        return None

    return _ResolvedStructRect(
        el=el,
        er=er,
        y0=x0 * lb_ratio,
        y1=x1 * lb_ratio,
        brightness_off=spec.brightness_off,
    )


def _consume_struct_geometry_draws(spec: Struct, rng: random.Random) -> None:
    rng.uniform(*spec.x0)
    rng.uniform(*spec.x1)
    rng.uniform(*spec.w)


def _estimate_struct_zone(
    spec: Struct,
    *,
    oversized_variant: bool = False,
) -> tuple[float, float]:
    x0 = spec.x0[0]
    x1 = spec.x1[1]
    if oversized_variant:
        x0, x1, _ = _apply_oversized_struct_geometry(x0, x1, spec.w[1])
    return x0, x1


def _write_struct_svg(
    out: StringIO,
    rect: _ResolvedStructRect,
    colors: ShipColors,
    rng: random.Random,
    sun_dx: float = 0.0,
) -> None:
    """Append one superstructure <rect>, clipped to hull width.

    Includes a one-side darkening overlay to simulate directional
    self-shadow on the structure face.
    """
    fill = colors.struct_css(rect.brightness_off, rng)
    stroke = colors.struct_css(rect.brightness_off - 25, rng)
    sw = min(rect.er - rect.el, rect.y1 - rect.y0) * 0.06
    out.write(
        f'  <rect x="{_f(rect.el)}" y="{_f(rect.y0)}" '
        f'width="{_f(rect.er - rect.el)}" height="{_f(rect.y1 - rect.y0)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{_f(sw)}" '
        f'data-role="struct"/>\n'
    )

    # Self-shadow: darken one face of the structure toward the sun direction.
    # The shadow colour is derived from the structure's own base colour so the
    # shaded face looks like the same material under reduced illumination rather
    # than a pure-black overlay.
    shadow_w = (rect.er - rect.el) * rng.uniform(0.25, 0.45)
    shadow_fill = colors.struct_shadow_css(rect.brightness_off, rng)
    if sun_dx > 0:
        # Sun from right → shadow on left side
        out.write(
            f'  <rect x="{_f(rect.el)}" y="{_f(rect.y0)}" '
            f'width="{_f(shadow_w)}" height="{_f(rect.y1 - rect.y0)}" '
            f'fill="{shadow_fill}" data-role="struct-shadow"/>\n'
        )
    else:
        # Sun from left → shadow on right side
        out.write(
            f'  <rect x="{_f(rect.er - shadow_w)}" y="{_f(rect.y0)}" '
            f'width="{_f(shadow_w)}" height="{_f(rect.y1 - rect.y0)}" '
            f'fill="{shadow_fill}" data-role="struct-shadow"/>\n'
        )


def _write_detail_svg(
    out: StringIO,
    detail: Detail,
    lb_ratio: float,
    colors: ShipColors,
    rng: random.Random,
) -> None:
    """Append SVG elements for one detail."""
    t = rng.uniform(*detail.x)
    cy = t * lb_ratio
    cx = detail.y
    sz = detail.size * lb_ratio
    ship_cx = 0.5
    kind = detail.kind

    if kind == "mast":
        w = sz * 0.3
        fill = colors.detail_css(-40)
        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - sz / 2)}" '
            f'width="{_f(w)}" height="{_f(sz)}" fill="{fill}"/>\n'
        )

    elif kind == "gun":
        r = sz * 0.5
        fill = colors.detail_css(-35)
        out.write(
            f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
            f'fill="{fill}"/>\n'
        )

    elif kind in ("helipad", "circle_spot"):
        r = sz * 0.5
        stroke = colors.detail_css(30)
        out.write(
            f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
            f'fill="none" stroke="{stroke}" stroke-width="{_f(sz * 0.08)}"/>\n'
        )

    elif kind == "vls":
        rows_n = max(2, int(sz / 0.02))
        cols = 2
        cell = sz / rows_n
        fill = colors.detail_css(10)
        stroke = colors.detail_css(-15)
        for ri in range(rows_n):
            for ci in range(cols):
                rx = cx - cols * cell / 2 + ci * cell
                ry = cy - rows_n * cell / 2 + ri * cell
                out.write(
                    f'  <rect x="{_f(rx)}" y="{_f(ry)}" '
                    f'width="{_f(cell * 0.9)}" height="{_f(cell * 0.9)}" '
                    f'fill="{fill}" stroke="{stroke}" '
                    f'stroke-width="{_f(cell * 0.05)}"/>\n'
                )

    elif kind == "crane":
        sw = sz * 0.12
        stroke = colors.detail_css(-35)
        # Vertical post
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy - sz)}" '
            f'x2="{_f(cx)}" y2="{_f(cy + sz)}" '
            f'stroke="{stroke}" stroke-width="{_f(sw)}"/>\n'
        )
        # Horizontal boom
        out.write(
            f'  <line x1="{_f(cx - sz * 0.5)}" y1="{_f(cy - sz)}" '
            f'x2="{_f(cx + sz * 0.5)}" y2="{_f(cy - sz)}" '
            f'stroke="{stroke}" stroke-width="{_f(sw)}"/>\n'
        )

    elif kind == "lamp":
        r = sz
        out.write(
            f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
            f'fill="rgb(210,210,195)"/>\n'
        )

    elif kind == "line":
        # Angled deck line (carriers)
        line_len = detail.size * lb_ratio
        x2 = cx + 0.35
        y2 = cy + line_len
        stroke = colors.detail_css(25)
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy)}" '
            f'x2="{_f(x2)}" y2="{_f(y2)}" '
            f'stroke="{stroke}" stroke-width="{_f(0.02)}"/>\n'
        )

    elif kind == "door":
        w = 0.6
        stroke = colors.detail_css(-30)
        out.write(
            f'  <line x1="{_f(ship_cx - w / 2)}" y1="{_f(cy)}" '
            f'x2="{_f(ship_cx + w / 2)}" y2="{_f(cy)}" '
            f'stroke="{stroke}" stroke-width="{_f(0.015)}"/>\n'
        )

    elif kind == "elevator":
        half = sz * 0.5
        stroke = colors.detail_css(10)
        out.write(
            f'  <rect x="{_f(cx - half)}" y="{_f(cy - half * 0.5)}" '
            f'width="{_f(half * 2)}" height="{_f(half)}" '
            f'fill="none" stroke="{stroke}" stroke-width="{_f(sz * 0.06)}"/>\n'
        )

    elif kind == "funnel":
        # Exhaust funnel — dark-topped rectangle
        w = sz * 0.7
        h = sz * 0.5
        fill = colors.detail_css(-25)
        top_fill = colors.detail_css(-45)
        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
            f'width="{_f(w)}" height="{_f(h)}" fill="{fill}"/>\n'
        )
        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
            f'width="{_f(w)}" height="{_f(h * 0.25)}" fill="{top_fill}"/>\n'
        )

    elif kind == "radar_dome":
        # Radome — light filled circle
        r = sz * 0.4
        fill = colors.detail_css(40)
        out.write(
            f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
            f'fill="{fill}"/>\n'
        )

    elif kind == "ciws":
        # Close-in weapon system — small dome with barrel
        r = sz * 0.35
        fill = colors.detail_css(-20)
        out.write(
            f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
            f'fill="{fill}"/>\n'
        )
        by = cy - r * 1.2
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy)}" '
            f'x2="{_f(cx)}" y2="{_f(by)}" '
            f'stroke="{fill}" stroke-width="{_f(sz * 0.08)}"/>\n'
        )

    elif kind == "winch":
        # Deck winch — small circle with stroke
        r = sz * 0.35
        fill = colors.detail_css(-15)
        stroke = colors.detail_css(-30)
        out.write(
            f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{_f(sz * 0.1)}"/>\n'
        )

    elif kind == "bollard":
        # Deck bollard — tiny filled rectangle
        w = sz * 0.3
        h = sz * 0.4
        fill = colors.detail_css(-30)
        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
            f'width="{_f(w)}" height="{_f(h)}" fill="{fill}"/>\n'
        )

    elif kind == "shadow":
        # Dark shadow cast by nearby structure.
        # Use a darkened, sky-ambient-tinted version of the hull colour rather
        # than pure black so the shadow looks like shaded deck surface.
        w = sz * 1.2
        h = sz * 0.6
        shadow_fill = colors.shadow_css(rng)
        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy)}" '
            f'width="{_f(w)}" height="{_f(h)}" '
            f'fill="{shadow_fill}" rx="{_f(h * 0.15)}"/>\n'
        )

    elif kind == "vent":
        # Ventilation grille — small square with darker inset
        s = sz * 0.6
        fill = colors.detail_css(-10)
        inner = colors.detail_css(-35)
        out.write(
            f'  <rect x="{_f(cx - s / 2)}" y="{_f(cy - s / 2)}" '
            f'width="{_f(s)}" height="{_f(s)}" fill="{fill}"/>\n'
        )
        si = s * 0.55
        out.write(
            f'  <rect x="{_f(cx - si / 2)}" y="{_f(cy - si / 2)}" '
            f'width="{_f(si)}" height="{_f(si)}" fill="{inner}"/>\n'
        )

    elif kind == "antenna":
        # Whip antenna — thin vertical line with small circle top
        h = sz * 1.5
        stroke = colors.detail_css(-40)
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy)}" '
            f'x2="{_f(cx)}" y2="{_f(cy - h)}" '
            f'stroke="{stroke}" stroke-width="{_f(sz * 0.06)}"/>\n'
        )
        out.write(
            f'  <circle cx="{_f(cx)}" cy="{_f(cy - h)}" r="{_f(sz * 0.08)}" '
            f'fill="{stroke}"/>\n'
        )

    elif kind == "davit":
        # Boat davit — small L-shaped arm
        arm = sz * 0.8
        stroke = colors.detail_css(-30)
        sw = sz * 0.1
        # Vertical post
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy)}" '
            f'x2="{_f(cx)}" y2="{_f(cy - arm)}" '
            f'stroke="{stroke}" stroke-width="{_f(sw)}"/>\n'
        )
        # Horizontal arm outward
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy - arm)}" '
            f'x2="{_f(cx + arm * 0.5)}" y2="{_f(cy - arm)}" '
            f'stroke="{stroke}" stroke-width="{_f(sw)}"/>\n'
        )

    elif kind == "pipe":
        # Exposed pipe run — thin horizontal line
        length = detail.size * lb_ratio * 2
        stroke = colors.detail_css(-22)
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy - length / 2)}" '
            f'x2="{_f(cx)}" y2="{_f(cy + length / 2)}" '
            f'stroke="{stroke}" stroke-width="{_f(sz * 0.15)}"/>\n'
        )

    elif kind == "liferaft":
        # Life raft canister — rounded rectangle
        w = sz * 0.7
        h = sz * 0.5
        fill = colors.detail_css(25)
        stroke = colors.detail_css(-5)
        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
            f'width="{_f(w)}" height="{_f(h)}" rx="{_f(h * 0.3)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{_f(sz * 0.06)}"/>\n'
        )

    elif kind == "tire_fender":
        # Side fender (tyre) — dark circle on hull edge
        r = sz * 0.4
        fill = colors.detail_css(-45)
        out.write(
            f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
            f'fill="{fill}"/>\n'
        )

    elif kind == "deck_line":
        # Deck marking / railing line along ship length
        length = detail.size * lb_ratio * 3
        stroke = colors.detail_css(-18)
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy - length / 2)}" '
            f'x2="{_f(cx)}" y2="{_f(cy + length / 2)}" '
            f'stroke="{stroke}" stroke-width="{_f(sz * 0.2)}" '
            f'stroke-dasharray="{_f(sz * 0.4)} {_f(sz * 0.2)}"/>\n'
        )

    elif kind == "hatch":
        # Cargo hatch — rectangle with inner cross lines
        w = sz * 0.8
        h = sz * 0.6
        fill = colors.detail_css(-8)
        stroke = colors.detail_css(-25)
        sw = sz * 0.04
        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
            f'width="{_f(w)}" height="{_f(h)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{_f(sw)}"/>\n'
        )
        # Centre cross
        out.write(
            f'  <line x1="{_f(cx - w / 2)}" y1="{_f(cy)}" '
            f'x2="{_f(cx + w / 2)}" y2="{_f(cy)}" '
            f'stroke="{stroke}" stroke-width="{_f(sw * 0.7)}"/>\n'
        )
        out.write(
            f'  <line x1="{_f(cx)}" y1="{_f(cy - h / 2)}" '
            f'x2="{_f(cx)}" y2="{_f(cy + h / 2)}" '
            f'stroke="{stroke}" stroke-width="{_f(sw * 0.7)}"/>\n'
        )


def _write_hull_tone_band(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    left_frac: float,
    right_frac: float,
    fill: str,
    *,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> None:
    """Draw a clipped tone band inside the hull using beam-relative fractions."""
    if left_frac >= right_frac:
        return

    n = len(half_widths)
    i_start = max(0, min(int(t_start * (n - 1)), n - 1))
    i_end = max(i_start, min(int(t_end * (n - 1)), n - 1))
    if i_end - i_start < 1:
        return

    left = max(-0.98, min(0.98, left_frac))
    right = max(-0.98, min(0.98, right_frac))
    pts: list[tuple[float, float]] = []
    for i in range(i_start, i_end + 1):
        t = i / (n - 1)
        y = t * lb_ratio
        hw = float(half_widths[i])
        pts.append((0.5 + hw * right, y))
    for i in range(i_end, i_start - 1, -1):
        t = i / (n - 1)
        y = t * lb_ratio
        hw = float(half_widths[i])
        pts.append((0.5 + hw * left, y))

    if len(pts) >= 3:
        out.write(
            f'  <polygon points="{_polygon_attr(pts)}" fill="{fill}"/>\n'
        )


def _side_interval(side_sign: int, inner_frac: float, outer_frac: float) -> tuple[float, float]:
    """Return a beam interval on the requested hull side."""
    inner = max(0.0, min(inner_frac, 0.98))
    outer = max(inner, min(outer_frac, 0.98))
    if side_sign < 0:
        return -outer, -inner
    return inner, outer


def _write_side_lighting(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    rng: random.Random,
    *,
    side_sign: int,
    bright: bool,
    opacity_range: tuple[float, float],
    outer_frac_range: tuple[float, float] = (0.84, 0.95),
    start_fracs: tuple[float, ...] = (0.12, 0.36, 0.60),
) -> None:
    """Approximate broadside lighting with several overlapping tone bands."""
    base_opacity = rng.uniform(*opacity_range)
    outer_frac = rng.uniform(*outer_frac_range)
    rgb = "255,255,255" if bright else "0,0,0"
    falloff = (0.52, 0.34, 0.22)

    for idx, start_frac in enumerate(start_fracs):
        if idx >= len(falloff):
            break
        inner_frac = min(start_frac + rng.uniform(-0.04, 0.04), outer_frac - 0.08)
        if inner_frac >= outer_frac - 0.04:
            continue
        left_frac, right_frac = _side_interval(side_sign, max(0.04, inner_frac), outer_frac)
        opacity = base_opacity * falloff[idx]
        t_start = rng.uniform(0.02, 0.10 + idx * 0.03)
        t_end = 1.0 - rng.uniform(0.02, 0.10 + idx * 0.03)
        _write_hull_tone_band(
            out,
            lb_ratio,
            half_widths,
            left_frac,
            right_frac,
            f"rgba({rgb},{opacity:.2f})",
            t_start=t_start,
            t_end=t_end,
        )


def _write_center_tone(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    rng: random.Random,
    *,
    bright: bool,
    preferred_side_sign: int | None = None,
    opacity_range: tuple[float, float] = (0.03, 0.08),
) -> None:
    """Draw a soft off-centre tone patch to avoid bilateral rim effects."""
    half_span = rng.uniform(0.24, 0.44)
    offset_sign = preferred_side_sign if preferred_side_sign is not None else rng.choice([-1, 1])
    offset = offset_sign * rng.uniform(0.12, 0.28)
    left_frac = max(-0.78, -half_span + offset)
    right_frac = min(0.78, half_span + offset)
    if right_frac - left_frac < 0.18:
        return
    rgb = "255,255,255" if bright else "0,0,0"
    opacity = rng.uniform(*opacity_range)
    _write_hull_tone_band(
        out,
        lb_ratio,
        half_widths,
        left_frac,
        right_frac,
        f"rgba({rgb},{opacity:.2f})",
        t_start=rng.uniform(0.05, 0.14),
        t_end=1.0 - rng.uniform(0.05, 0.14),
    )


def _write_hull_mottling(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    rng: random.Random,
) -> None:
    """Overlay large irregular patches of slight brightness variation.

    Breaks up the flat hull colour into 3-6 tonal zones, giving the
    appearance of paint wear, panel boundaries, and deck surface
    variation that are visible even in low-resolution satellite imagery.
    """
    import math

    n = len(half_widths)

    # ── Large tonal patches (3-6) ────────────────────────────────────
    n_patches = rng.randint(3, 6)

    for _ in range(n_patches):
        # Random centre on the hull
        ct = rng.uniform(0.10, 0.90)
        idx = min(int(ct * (n - 1)), n - 1)
        hw = float(half_widths[idx])
        if hw < 0.05:
            continue
        cy = ct * lb_ratio
        cx = 0.5 + rng.uniform(-hw * 0.5, hw * 0.5)

        # Blob radius relative to hull size
        r_y = rng.uniform(0.08, 0.25) * lb_ratio
        r_x = rng.uniform(0.06, 0.18)

        # Build irregular blob polygon (8 vertices)
        n_verts = 8
        pts: list[tuple[float, float]] = []
        for vi in range(n_verts):
            angle = 2 * math.pi * vi / n_verts
            jitter = rng.uniform(0.7, 1.3)
            px = cx + r_x * jitter * math.cos(angle)
            py = cy + r_y * jitter * math.sin(angle)
            pts.append((px, py))

        # Randomly lighten or darken
        if rng.random() < 0.5:
            opacity = rng.uniform(0.06, 0.15)
            fill = f"rgba(0,0,0,{opacity:.2f})"
        else:
            opacity = rng.uniform(0.05, 0.13)
            fill = f"rgba(255,255,255,{opacity:.2f})"

        out.write(
            f'  <polygon points="{_polygon_attr(pts)}" fill="{fill}"/>\n'
        )

    # ── Fine-scale surface noise (many small patches) ────────────────
    # Dense layer of small, very faint marks that provide overall
    # surface texture — simulates paint grain, minor irregularities.
    n_fine = rng.randint(max(4, int(lb_ratio * 2)), max(6, int(lb_ratio * 4)))
    for _ in range(n_fine):
        ct = rng.uniform(0.04, 0.96)
        idx = min(int(ct * (n - 1)), n - 1)
        hw = float(half_widths[idx])
        if hw < 0.04:
            continue
        cy = ct * lb_ratio
        cx = 0.5 + rng.uniform(-hw * 0.7, hw * 0.7)

        # Small blobs: 4-6 vertices
        r_y = rng.uniform(0.02, 0.06) * lb_ratio
        r_x = rng.uniform(0.015, 0.05)
        n_verts = rng.randint(4, 6)
        pts = []
        for vi in range(n_verts):
            angle = 2 * math.pi * vi / n_verts
            jitter = rng.uniform(0.6, 1.4)
            px = cx + r_x * jitter * math.cos(angle)
            py = cy + r_y * jitter * math.sin(angle)
            pts.append((px, py))

        opacity = rng.uniform(0.03, 0.07)
        if rng.random() < 0.55:
            fill = f"rgba(0,0,0,{opacity:.2f})"
        else:
            fill = f"rgba(255,255,255,{opacity:.2f})"
        out.write(
            f'  <polygon points="{_polygon_attr(pts)}" fill="{fill}"/>\n'
        )


def _write_bow_stern_shading(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    rng: random.Random,
) -> None:
    """Darken bow and stern extremities to simulate curvature shading.

    Real vessels appear darker near the narrowing bow/stern due to
    surface angle relative to satellite view and paint weathering.
    """
    n = len(half_widths)
    bow_len = rng.uniform(0.10, 0.18)  # fraction of total length
    stern_len = rng.uniform(0.10, 0.22)
    bow_opacity = rng.uniform(0.08, 0.18)
    stern_opacity = rng.uniform(0.06, 0.15)

    # Bow zone — multiple strips with decreasing opacity (gradient approx)
    n_strips = 4
    for si in range(n_strips):
        t_end = bow_len * (n_strips - si) / n_strips
        strip_opacity = bow_opacity * (si + 1) / n_strips
        pts: list[tuple[float, float]] = []
        i_end = min(int(t_end * (n - 1)), n - 1) + 1
        for i in range(i_end):
            t = i / (n - 1)
            y = t * lb_ratio
            hw = float(half_widths[i])
            pts.append((0.5 + hw, y))
        for i in range(i_end - 1, -1, -1):
            t = i / (n - 1)
            y = t * lb_ratio
            hw = float(half_widths[i])
            pts.append((0.5 - hw, y))
        if len(pts) >= 3:
            out.write(
                f'  <polygon points="{_polygon_attr(pts)}" '
                f'fill="rgba(0,0,0,{strip_opacity:.2f})"/>\n'
            )

    # Stern zone
    for si in range(n_strips):
        t_start = 1.0 - stern_len * (n_strips - si) / n_strips
        strip_opacity = stern_opacity * (si + 1) / n_strips
        pts = []
        i_start = max(int(t_start * (n - 1)), 0)
        for i in range(i_start, n):
            t = i / (n - 1)
            y = t * lb_ratio
            hw = float(half_widths[i])
            pts.append((0.5 + hw, y))
        for i in range(n - 1, i_start - 1, -1):
            t = i / (n - 1)
            y = t * lb_ratio
            hw = float(half_widths[i])
            pts.append((0.5 - hw, y))
        if len(pts) >= 3:
            out.write(
                f'  <polygon points="{_polygon_attr(pts)}" '
                f'fill="rgba(0,0,0,{strip_opacity:.2f})"/>\n'
            )


def _write_deck_panels(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    colors: ShipColors,
    rng: random.Random,
    struct_zones: list[tuple[float, float]] | None = None,
) -> None:
    """Draw large-scale deck panel divisions and centreline markings.

    Adds the "big structural" appearance — panel seam lines running
    across the deck, longitudinal centreline, and occasional large
    rectangular zone fills that break the hull into distinct deck areas.
    These are the coarse features most visible in satellite imagery.
    """
    n = len(half_widths)
    sz_zones = struct_zones or []

    # ── Centreline ───────────────────────────────────────────────────
    # Thin darker/lighter line running bow to stern along the midship axis
    cl_opacity = rng.uniform(0.08, 0.16)
    cl_bright = rng.choice([True, False])
    cl_colour = f"rgba(255,255,255,{cl_opacity:.2f})" if cl_bright else f"rgba(0,0,0,{cl_opacity:.2f})"
    cl_hw = rng.uniform(0.016, 0.028)
    # Build a thin strip polygon
    cl_pts: list[tuple[float, float]] = []
    for i in range(n):
        t = i / (n - 1)
        y = t * lb_ratio
        cl_pts.append((0.5 - cl_hw, y))
    for i in range(n - 1, -1, -1):
        t = i / (n - 1)
        y = t * lb_ratio
        cl_pts.append((0.5 + cl_hw, y))
    out.write(
        f'  <polygon points="{_polygon_attr(cl_pts)}" fill="{cl_colour}"/>\n'
    )

    # ── Transverse seam lines ────────────────────────────────────────
    # Horizontal lines across the deck at irregular intervals
    n_seams = rng.randint(3, max(4, int(lb_ratio * 1.2)))
    seam_opacity = rng.uniform(0.07, 0.14)
    for _ in range(n_seams):
        t = rng.uniform(0.08, 0.92)
        # Skip if inside a structure zone
        in_struct = any(zs <= t <= ze for zs, ze in sz_zones)
        if in_struct:
            continue
        idx = min(int(t * (n - 1)), n - 1)
        hw = float(half_widths[idx])
        if hw < 0.05:
            continue
        y = t * lb_ratio
        seam_h = rng.uniform(0.008, 0.018) * lb_ratio
        out.write(
            f'  <rect x="{_f(0.5 - hw * 0.92)}" y="{_f(y)}" '
            f'width="{_f(hw * 1.84)}" height="{_f(seam_h)}" '
            f'fill="rgba(0,0,0,{seam_opacity:.2f})"/>\n'
        )

    # ── Large deck zone fills ────────────────────────────────────────
    # 1-3 large rectangles covering different deck areas with subtle
    # tone differences, simulating different deck surface materials.
    n_zones = rng.randint(1, 3)
    for _ in range(n_zones):
        t0 = rng.uniform(0.10, 0.70)
        t1 = t0 + rng.uniform(0.08, 0.25)
        if t1 > 0.95:
            continue
        # Skip if overlapping structures
        in_struct = any(
            not (t1 < zs or t0 > ze) for zs, ze in sz_zones
        )
        if in_struct:
            continue
        idx0 = min(int(t0 * (n - 1)), n - 1)
        idx1 = min(int(t1 * (n - 1)), n - 1)
        # Use narrower than full hull width for panel effect
        w_frac = rng.uniform(0.6, 0.90)
        zone_pts: list[tuple[float, float]] = []
        for i in range(idx0, idx1 + 1):
            ti = i / (n - 1)
            y = ti * lb_ratio
            hw = float(half_widths[i])
            zone_pts.append((0.5 + hw * w_frac, y))
        for i in range(idx1, idx0 - 1, -1):
            ti = i / (n - 1)
            y = ti * lb_ratio
            hw = float(half_widths[i])
            zone_pts.append((0.5 - hw * w_frac, y))
        if len(zone_pts) >= 3:
            zone_opacity = rng.uniform(0.04, 0.10)
            if rng.random() < 0.5:
                fill = f"rgba(0,0,0,{zone_opacity:.2f})"
            else:
                fill = f"rgba(255,255,255,{zone_opacity:.2f})"
            out.write(
                f'  <polygon points="{_polygon_attr(zone_pts)}" fill="{fill}"/>\n'
            )


def _write_deck_wear(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    rng: random.Random,
    struct_zones: list[tuple[float, float]] | None = None,
) -> None:
    """Add low-frequency deck surface wear and stain marks.

    Larger and softer than deck scatter — simulates paint wear, rust
    streaks, and surface discoloration visible from satellite altitude.
    """
    n = len(half_widths)
    sz_zones = struct_zones or []
    n_stains = rng.randint(2, max(3, int(lb_ratio * 0.8)))

    for _ in range(n_stains):
        t = rng.uniform(0.06, 0.94)
        in_struct = any(zs <= t <= ze for zs, ze in sz_zones)
        if in_struct:
            continue
        idx = min(int(t * (n - 1)), n - 1)
        hw = float(half_widths[idx])
        if hw < 0.05:
            continue
        cy = t * lb_ratio
        cx = 0.5 + rng.uniform(-hw * 0.6, hw * 0.6)

        # Elliptical or rectangular stain
        w = rng.uniform(0.04, 0.12)
        h = rng.uniform(0.03, 0.10) * lb_ratio
        opacity = rng.uniform(0.05, 0.14)

        if rng.random() < 0.6:
            # Dark stain (wear, oil, dirt)
            fill = f"rgba(0,0,0,{opacity:.2f})"
        else:
            # Light residue
            fill = f"rgba(255,255,255,{opacity * 0.8:.2f})"

        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
            f'width="{_f(w)}" height="{_f(h)}" '
            f'fill="{fill}" rx="{_f(min(w, h) * 0.3)}"/>\n'
        )


def _write_struct_shadow_svg(
    out: StringIO,
    rect: _ResolvedStructRect,
    sun_dx: float,
    sun_dy: float,
    colors: ShipColors,
    rng: random.Random,
) -> None:
    """Draw a cast shadow offset from a superstructure block.

    The shadow colour is derived from the hull colour — darkened and
    sky-ambient-tinted — rather than pure black.  The element is a solid
    opaque rectangle; overall ship transparency is set externally.
    """
    y0 = rect.y0 + sun_dy
    y1 = rect.y1 + sun_dy
    el_s = rect.el + sun_dx
    er_s = rect.er + sun_dx
    shadow_fill = colors.shadow_css(rng)
    out.write(
        f'  <rect x="{_f(el_s)}" y="{_f(y0)}" '
        f'width="{_f(er_s - el_s)}" height="{_f(y1 - y0)}" '
        f'fill="{shadow_fill}" rx="{_f((y1 - y0) * 0.05)}"/>\n'
    )


def _write_deck_scatter_svg(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    colors: ShipColors,
    rng: random.Random,
    density: float = 3.0,
    struct_zones: list[tuple[float, float]] | None = None,
    sun_dx: float = 0.0,
    sun_dy: float = 0.0,
) -> None:
    """甲板上にランダムな小図形を散布してテクスチャを付加する。

    hull の <clipPath> 内部に包んだ <g> から呼び出すこと。

    Parameters
    ----------
    density
        散布密度。1 あたりの図形数 = density * lb_ratio。
        0 で散布なし。標準は 3.0。
    struct_zones
        Superstructure exclusion zones as ``(t_start, t_end)`` along the
        normalised ship length [0, 1].  Scatter points falling inside
        these regions are skipped.
    sun_dx, sun_dy
        Shadow offset direction shared with superstructure shadows.
    """
    if density <= 0:
        return

    n = len(half_widths)
    base = density * lb_ratio
    n_shapes = rng.randint(max(1, int(base * 0.6)), max(1, int(base * 1.4)) + 1)

    sz_zones = struct_zones or []

    for _ in range(n_shapes):
        t = rng.uniform(0.04, 0.96)

        # Skip if this point falls inside a superstructure zone
        in_struct = False
        for z_start, z_end in sz_zones:
            if z_start <= t <= z_end:
                in_struct = True
                break
        if in_struct:
            continue

        idx = min(int(t * (n - 1)), n - 1)
        hull_hw = float(half_widths[idx])
        if hull_hw < 0.05:
            continue

        cy = t * lb_ratio
        margin = min(0.04, hull_hw * 0.15)
        half_range = hull_hw - margin
        cx = 0.5 + rng.uniform(-half_range, half_range)

        # ── Size tier selection ──────────────────────────────────────
        # Tier 1 (45%): micro texture — stains, tie-downs, drains
        # Tier 2 (35%): small equipment — vents, reels, foundations
        # Tier 3 (20%): medium equipment — boxes, hatches, containers
        #
        # Size targets are chosen so shapes survive Gaussian blur σ≤2 px:
        #   survive threshold ≈ 3σ = 6 px at beam_px=60 (preview minimum).
        #   Tier 1 max 0.120 →  7.2 px @ 60 px  (barely visible, intentional texture)
        #   Tier 2 max 0.180 → 10.8 px @ 60 px  (visible detail)
        #   Tier 3 max 0.300 → 18.0 px @ 60 px  (clearly visible structure)
        tier_roll = rng.random()
        if tier_roll < 0.45:
            tier = 1
            max_sz = min(0.120, hull_hw * 0.30)
            sz = rng.uniform(max_sz * 0.50, max_sz)
        elif tier_roll < 0.80:
            tier = 2
            max_sz = min(0.180, hull_hw * 0.50)
            sz = rng.uniform(max_sz * 0.50, max_sz)
        else:
            tier = 3
            max_sz = min(0.300, hull_hw * 0.75)
            sz = rng.uniform(max_sz * 0.55, max_sz)

        # ── Colour — hull-toned with strong contrast ─────────────────
        # Contrast values represent Δ in [0,255] per RGB channel.
        # Real deck equipment reads as Δ≥30 (rust/grime) to Δ≥80 (white hatches).
        # Values are clamped to [0,255] inside detail_css().
        sign = 1 if rng.random() < 0.5 else -1
        contrast = {1: (30, 65), 2: (45, 80), 3: (65, 110)}[tier]
        offset = sign * rng.randint(*contrast)
        fill = colors.detail_css(offset)

        # ── Shadow for tier 2+ ──────────────────────────────────────
        if tier >= 2:
            shadow_dx = sun_dx * 0.4
            shadow_dy = sun_dy * 0.4
            _write_scatter_shadow(
                out, cx, cy, sz, shadow_dx, shadow_dy, colors, rng,
            )

        # ── Shape drawing ────────────────────────────────────────────
        sw = sz * 0.06

        if tier == 1:
            # Micro marks: tiny rects and dots — no stroke
            kind = rng.choices(["dot", "tick_v", "tick_h"], weights=[3, 2, 2])[0]
            if kind == "dot":
                r = sz * 0.4
                out.write(
                    f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
                    f'fill="{fill}"/>\n'
                )
            elif kind == "tick_v":
                hl = sz * rng.uniform(0.6, 1.2)
                out.write(
                    f'  <line x1="{_f(cx)}" y1="{_f(cy - hl)}" '
                    f'x2="{_f(cx)}" y2="{_f(cy + hl)}" '
                    f'stroke="{fill}" stroke-width="{_f(sz * 0.25)}"/>\n'
                )
            else:  # tick_h
                hw = sz * rng.uniform(0.5, 1.0)
                out.write(
                    f'  <line x1="{_f(cx - hw)}" y1="{_f(cy)}" '
                    f'x2="{_f(cx + hw)}" y2="{_f(cy)}" '
                    f'stroke="{fill}" stroke-width="{_f(sz * 0.25)}"/>\n'
                )
        elif tier == 2:
            # Small equipment: rects aligned to ship axis, circles
            stroke_color = colors.detail_css(offset - sign * 6)
            kind = rng.choices(
                ["rect", "rect_long", "circle"],
                weights=[4, 3, 2],
            )[0]
            if kind == "rect":
                w = sz * rng.uniform(0.6, 1.2)
                h = sz * rng.uniform(0.6, 1.2)
                out.write(
                    f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
                    f'width="{_f(w)}" height="{_f(h)}" '
                    f'fill="{fill}" stroke="{stroke_color}" '
                    f'stroke-width="{_f(sw)}"/>\n'
                )
            elif kind == "rect_long":
                w = sz * rng.uniform(0.3, 0.6)
                h = sz * rng.uniform(1.5, 2.5)
                out.write(
                    f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
                    f'width="{_f(w)}" height="{_f(h)}" '
                    f'fill="{fill}" stroke="{stroke_color}" '
                    f'stroke-width="{_f(sw)}"/>\n'
                )
            else:  # circle
                r = sz * 0.45
                out.write(
                    f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
                    f'fill="{fill}" stroke="{stroke_color}" '
                    f'stroke-width="{_f(sw)}"/>\n'
                )
        else:
            # Tier 3: medium equipment — boxes, small hatches with crosslines
            stroke_color = colors.detail_css(offset - sign * 8)
            kind = rng.choices(
                ["box", "hatch_mini", "rect_long"],
                weights=[4, 3, 2],
            )[0]
            if kind == "box":
                w = sz * rng.uniform(0.7, 1.3)
                h = sz * rng.uniform(0.7, 1.3)
                out.write(
                    f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
                    f'width="{_f(w)}" height="{_f(h)}" '
                    f'fill="{fill}" stroke="{stroke_color}" '
                    f'stroke-width="{_f(sw)}"/>\n'
                )
            elif kind == "hatch_mini":
                w = sz * rng.uniform(0.8, 1.2)
                h = sz * rng.uniform(0.6, 1.0)
                inner = colors.detail_css(offset - sign * 14)
                out.write(
                    f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
                    f'width="{_f(w)}" height="{_f(h)}" '
                    f'fill="{fill}" stroke="{stroke_color}" '
                    f'stroke-width="{_f(sw)}"/>\n'
                )
                # Cross divider
                out.write(
                    f'  <line x1="{_f(cx - w / 2)}" y1="{_f(cy)}" '
                    f'x2="{_f(cx + w / 2)}" y2="{_f(cy)}" '
                    f'stroke="{inner}" stroke-width="{_f(sw * 0.6)}"/>\n'
                )
            else:  # rect_long
                w = sz * rng.uniform(0.3, 0.5)
                h = sz * rng.uniform(2.0, 3.5)
                out.write(
                    f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
                    f'width="{_f(w)}" height="{_f(h)}" '
                    f'fill="{fill}" stroke="{stroke_color}" '
                    f'stroke-width="{_f(sw)}"/>\n'
                )


def _write_scatter_shadow(
    out: StringIO,
    cx: float,
    cy: float,
    sz: float,
    shadow_dx: float,
    shadow_dy: float,
    colors: ShipColors,
    rng: random.Random,
) -> None:
    """Render a tiny drop shadow for one deck scatter item."""
    sx = cx + shadow_dx
    sy = cy + shadow_dy
    w = sz * rng.uniform(0.8, 1.3)
    h = sz * rng.uniform(0.8, 1.3)
    shadow_fill = colors.shadow_css(rng)
    out.write(
        f'  <rect x="{_f(sx - w / 2)}" y="{_f(sy - h / 2)}" '
        f'width="{_f(w)}" height="{_f(h)}" '
        f'fill="{shadow_fill}" '
        f'rx="{_f(min(w, h) * 0.15)}"/>\n'
    )


# ── Public API ───────────────────────────────────────────────────────────


def get_ship_classes(*, include_debug: bool = False) -> list[str]:
    """Return sorted list of available ship class names."""
    if include_debug:
        return sorted(SHIP_CLASSES)
    return sorted(name for name, cls in SHIP_CLASSES.items() if not cls.debug_only)


def generate_ship_svg(
    ship_class: str,
    *,
    rng: random.Random | None = None,
    hull_noise: float = 0.005,
    n_hull_points: int = 64,
    deck_scatter_density: float = 3.0,
    trim_mode: str | None = None,
    offnadir_deg: float = 0.0,
    sensor_az_ship_deg: float = 0.0,
) -> str:
    """Generate a single ship as an SVG string.

    The SVG viewBox is ``0 0 1 {lb_ratio}`` — beam is normalised to 1,
    length equals the L/B ratio.  Bow is at y = 0.

    Parameters
    ----------
    ship_class
        Key in :data:`SHIP_CLASSES`.
    rng
        Random state.  Created from system entropy when *None*.
    hull_noise
        Standard deviation of hull outline perturbation.
    n_hull_points
        Number of polygon sample points per side.
    deck_scatter_density
        Scatter shape density on deck.  Shapes per unit lb_ratio.
        0 disables scatter entirely.  Default is 3.0.
    trim_mode
        Optional forced hull trim mode: ``none``, ``perimeter``, or ``bow``.
        ``None`` samples from the class family defaults.
    offnadir_deg
        Off-nadir viewing angle in degrees (0 = nadir/overhead).  Controls
        how much of the ship's side is visible.  Must be >= 0.
    sensor_az_ship_deg
        Sensor azimuth in ship frame (degrees).  0 = bow-on, 90 = looking at
        starboard side, 180 = stern-on, 270 = looking at port side.  Together
        with *offnadir_deg* this determines which side is visible and how wide
        the side band appears: ``side_component = tan(offnadir) * sin(az)``.

    Returns
    -------
    str
        Well-formed SVG document.
    """
    if rng is None:
        rng = random.Random()

    cls = SHIP_CLASSES[ship_class]

    # Sample per-instance parameters
    lb_ratio = rng.uniform(*cls.lb)
    bow_sharpness = rng.uniform(*cls.bow)
    stern_hw = rng.uniform(*cls.stern_hw)
    appearance_variant = sample_ship_appearance_variant(cls, rng)

    if ship_class == "debug_rect":
        hull_pts = _debug_rect_points(lb_ratio)
        fill = _rgb_css(rng.choice(_DEBUG_RECT_COLORS))

        out = StringIO()
        out.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 1 {_f(lb_ratio)}" '
            f'data-ship-class="{ship_class}" '
            f'data-trim-mode="none" '
            f'data-visible-side="none" '
            f'data-lb-ratio="{_f(lb_ratio)}">\n'
        )
        out.write(
            f'  <defs>\n'
            f'    <clipPath id="h">\n'
            f'      <polygon points="{_polygon_attr(hull_pts)}"/>\n'
            f'    </clipPath>\n'
            f'  </defs>\n'
        )
        out.write(
            f'  <polygon points="{_polygon_attr(hull_pts)}" '
            f'fill="{fill}"/>\n'
        )
        out.write("</svg>\n")
        return out.getvalue()

    # Compute off-nadir viewing geometry.
    # side_component > 0 → starboard visible; < 0 → port visible; ≈ 0 → none.
    tan_theta = math.tan(math.radians(offnadir_deg))
    side_component = tan_theta * math.sin(math.radians(sensor_az_ship_deg))
    beam_shift = side_component * _STRUCT_HEIGHT_FRAC

    colors = sample_colors(
        cls.color_family,
        rng,
        trim_mode=trim_mode,
        side_component=side_component,
        appearance_variant=appearance_variant,
    )

    # Hull outline in normalised coords
    half_widths = interpolate_hull(
        cls.hull, bow_sharpness, stern_hw, n_hull_points,
    )
    hull_pts = build_hull_points(half_widths, lb_ratio, rng, hull_noise)

    # Build SVG
    out = StringIO()
    out.write(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 1 {_f(lb_ratio)}" '
        f'data-ship-class="{ship_class}" '
        f'data-trim-mode="{colors.trim.primary_mode}" '
        f'data-visible-side="{colors.trim.visible_side}" '
        f'data-lb-ratio="{_f(lb_ratio)}">\n'
    )

    # ClipPath — used to constrain deck scatter and edge effects to hull
    out.write(
        f'  <defs>\n'
        f'    <clipPath id="h">\n'
        f'      <polygon points="{_polygon_attr(hull_pts)}"/>\n'
        f'    </clipPath>\n'
        f'  </defs>\n'
    )

    # 1) Hull polygon
    out.write(
        f'  <polygon points="{_polygon_attr(hull_pts)}" '
        f'fill="{colors.hull_css()}"/>\n'
    )
    _write_hull_trim(out, hull_pts, colors)

    # 2) Superstructures (with directional shadow)
    # Pick a consistent sun angle for the whole ship
    sun_dx = rng.uniform(-0.02, 0.02)
    sun_dy = rng.uniform(0.01, 0.04) * lb_ratio  # shadow falls roughly aft/side

    # ── Shading style selection ──────────────────────────────────────────
    # Avoid a bilateral edge outline.  Ships can be flat, softly side-lit,
    # or show a glancing highlight on one side, but should not systematically
    # produce dark or bright rims on both edges.
    shading_style: str = rng.choices(
        ["flat", "broadside_shadow", "broadside_light", "center_tone", "edge_glint"],
        weights=[18, 30, 24, 16, 12],
    )[0]
    sun_side_sign = 1 if sun_dx > 0 else -1
    shadow_side_sign = -sun_side_sign

    # 1b) Primary hull lighting
    out.write('  <g clip-path="url(#h)">\n')
    if shading_style == "broadside_shadow":
        _write_side_lighting(
            out,
            lb_ratio,
            half_widths,
            rng,
            side_sign=shadow_side_sign,
            bright=False,
            opacity_range=(0.08, 0.16),
        )
    elif shading_style == "broadside_light":
        _write_side_lighting(
            out,
            lb_ratio,
            half_widths,
            rng,
            side_sign=sun_side_sign,
            bright=True,
            opacity_range=(0.06, 0.13),
        )
    elif shading_style == "center_tone":
        _write_center_tone(
            out,
            lb_ratio,
            half_widths,
            rng,
            bright=rng.random() < 0.55,
            preferred_side_sign=sun_side_sign,
        )
    elif shading_style == "edge_glint":
        _write_side_lighting(
            out,
            lb_ratio,
            half_widths,
            rng,
            side_sign=sun_side_sign,
            bright=True,
            opacity_range=(0.05, 0.10),
            outer_frac_range=(0.90, 0.96),
            start_fracs=(0.66, 0.78),
        )
    out.write('  </g>\n')

    # 1c) Secondary lighting counterpart
    out.write('  <g clip-path="url(#h)">\n')
    if shading_style == "broadside_shadow":
        _write_side_lighting(
            out,
            lb_ratio,
            half_widths,
            rng,
            side_sign=sun_side_sign,
            bright=True,
            opacity_range=(0.03, 0.07),
            outer_frac_range=(0.68, 0.88),
            start_fracs=(0.18, 0.42),
        )
    elif shading_style == "broadside_light":
        _write_side_lighting(
            out,
            lb_ratio,
            half_widths,
            rng,
            side_sign=shadow_side_sign,
            bright=False,
            opacity_range=(0.03, 0.07),
            outer_frac_range=(0.70, 0.88),
            start_fracs=(0.16, 0.40),
        )
    elif shading_style == "center_tone" and rng.random() < 0.45:
        _write_side_lighting(
            out,
            lb_ratio,
            half_widths,
            rng,
            side_sign=sun_side_sign,
            bright=rng.random() < 0.5,
            opacity_range=(0.02, 0.05),
            outer_frac_range=(0.64, 0.84),
            start_fracs=(0.22, 0.44),
        )
    elif shading_style == "edge_glint":
        _write_side_lighting(
            out,
            lb_ratio,
            half_widths,
            rng,
            side_sign=shadow_side_sign,
            bright=False,
            opacity_range=(0.02, 0.05),
            outer_frac_range=(0.60, 0.78),
            start_fracs=(0.30, 0.50),
        )
    out.write('  </g>\n')

    # 1d) Hull colour mottling — large irregular patches break up the
    # flat base colour into multiple tonal zones.
    out.write('  <g clip-path="url(#h)">\n')
    _write_hull_mottling(out, lb_ratio, half_widths, rng)
    out.write('  </g>\n')

    # 1e) Bow / stern gradient shading — darken toward extremities
    out.write('  <g clip-path="url(#h)">\n')
    _write_bow_stern_shading(out, lb_ratio, half_widths, rng)
    out.write('  </g>\n')

    # 1f) Very soft hull-side asymmetry shared across all styles
    out.write('  <g clip-path="url(#h)">\n')
    _write_side_lighting(
        out,
        lb_ratio,
        half_widths,
        rng,
        side_sign=shadow_side_sign,
        bright=False,
        opacity_range=(0.02, 0.05),
        outer_frac_range=(0.62, 0.82),
        start_fracs=(0.26, 0.50),
    )
    out.write('  </g>\n')

    # Compute approximate superstructure exclusion zones early (used by
    # panel divisions, deck wear, and scatter).
    struct_zones = [
        _estimate_struct_zone(
            spec,
            oversized_variant=appearance_variant.oversized_struct and index == 0,
        )
        for index, spec in enumerate(cls.structs)
    ]

    # 1g) Deck panel divisions — centreline, seam lines, zone fills
    out.write('  <g clip-path="url(#h)">\n')
    _write_deck_panels(out, lb_ratio, half_widths, colors, rng, struct_zones)
    out.write('  </g>\n')

    # 1h) Deck surface wear — low-frequency stains and discoloration
    out.write('  <g clip-path="url(#h)">\n')
    _write_deck_wear(out, lb_ratio, half_widths, rng, struct_zones)
    out.write('  </g>\n')

    for index, s in enumerate(cls.structs):
        if rng.random() < s.prob:
            rect = _resolve_struct_rect(
                s,
                lb_ratio,
                half_widths,
                rng,
                oversized_variant=appearance_variant.oversized_struct and index == 0,
                beam_shift=beam_shift,
            )
            if rect is None:
                _consume_struct_geometry_draws(s, rng)
                continue
            _write_struct_shadow_svg(out, rect, sun_dx, sun_dy, colors, rng)
            _consume_struct_geometry_draws(s, rng)
            _write_struct_svg(out, rect, colors, rng, sun_dx)

    # 3) Deck scatter — random small shapes clipped to hull for visual texture
    out.write('  <g clip-path="url(#h)" id="scatter">\n')
    _write_deck_scatter_svg(
        out, lb_ratio, half_widths, colors, rng, deck_scatter_density,
        struct_zones=struct_zones, sun_dx=sun_dx, sun_dy=sun_dy,
    )
    out.write('  </g>\n')

    # 4) Details
    for d in cls.details:
        if rng.random() < d.prob:
            _write_detail_svg(out, d, lb_ratio, colors, rng)

    out.write("</svg>\n")
    return out.getvalue()


def _svg_to_png_bytes(svg: str, width_px: int = 64) -> bytes:
    """Rasterize a shipgen SVG to PNG bytes using the existing render pipeline.

    Parameters
    ----------
    svg
        SVG text produced by :func:`generate_ship_svg`.
    width_px
        Output width in pixels; height is derived from the viewBox aspect ratio.
    """
    import xml.etree.ElementTree as _ET

    import numpy as np
    from PIL import Image

    from medetect.datagen.render import rasterize_ship_svg

    root = _ET.fromstring(svg)
    vb = root.get("viewBox", "0 0 1 1").split()
    vb_w, vb_h = float(vb[2]), float(vb[3])
    height_px = max(1, round(width_px * vb_h / vb_w)) if vb_w > 0 else width_px

    rgba: np.ndarray = rasterize_ship_svg(svg, width_px=width_px, height_px=height_px)
    img = Image.fromarray(rgba, mode="RGBA")

    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_ships(
    output_dir: Path,
    count: int,
    *,
    types: dict[str, float] | None = None,
    seed: int | None = None,
    hull_noise: float = 0.005,
    n_hull_points: int = 64,
    deck_scatter_density: float = 3.0,
    filetype: str = "svg",
    offnadir_max: float = 0.0,
) -> None:
    """Generate synthetic ship files.

    Parameters
    ----------
    output_dir
        Destination directory (created if absent).
    count
        Number of files to generate.
    types
        ``{ship_class: weight}`` mapping.  Equal weights for all classes
        when *None*.
    seed
        Random seed for reproducibility.
    hull_noise
        Standard deviation of hull outline perturbation.
    n_hull_points
        Number of polygon sample points per side.
    deck_scatter_density
        Scatter shape density on deck passed to :func:`generate_ship_svg`.
    filetype
        Output format: ``"svg"`` (default) or ``"png"``.
    offnadir_max
        Maximum off-nadir angle in degrees (0 = nadir only).  Each ship
        independently draws ``offnadir_deg ~ Uniform(0, offnadir_max)`` and
        ``sensor_az_ship_deg ~ Uniform(0, 360)``.

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
                msg = f"Unknown ship class: {name!r}. Available: {get_ship_classes(include_debug=True)}"
                raise ValueError(msg)
        classes = list(types)
        weights = [types[c] for c in classes]

    counters: dict[str, int] = {}

    for i in range(count):
        ship_class = rng.choices(classes, weights=weights, k=1)[0]
        offnadir_deg = rng.uniform(0.0, offnadir_max)
        sensor_az_ship_deg = rng.uniform(0.0, 360.0)
        svg = generate_ship_svg(
            ship_class, rng=rng, hull_noise=hull_noise,
            n_hull_points=n_hull_points,
            deck_scatter_density=deck_scatter_density,
            offnadir_deg=offnadir_deg,
            sensor_az_ship_deg=sensor_az_ship_deg,
        )

        counters[ship_class] = counters.get(ship_class, 0) + 1
        if filetype == "png":
            filename = f"{ship_class}_{counters[ship_class]:05d}.png"
            (output_dir / filename).write_bytes(_svg_to_png_bytes(svg))
        else:
            filename = f"{ship_class}_{counters[ship_class]:05d}.svg"
            (output_dir / filename).write_text(svg, encoding="utf-8")

        if (i + 1) % 100 == 0 or (i + 1) == count:
            logger.info("Generated %d / %d %s files", i + 1, count, filetype.upper())

    logger.info("Ship counts: %s", dict(sorted(counters.items())))
