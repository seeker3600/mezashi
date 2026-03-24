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
import random
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
)

logger = logging.getLogger(__name__)

# ── SVG formatting helpers ───────────────────────────────────────────────


def _f(v: float) -> str:
    """Format a float for SVG coordinates (4 decimal places, strip trailing zeros)."""
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _polygon_attr(pts: list[tuple[float, float]]) -> str:
    return " ".join(f"{_f(x)},{_f(y)}" for x, y in pts)


# ── SVG element writers ──────────────────────────────────────────────────


def _write_struct_svg(
    out: StringIO,
    spec: Struct,
    lb_ratio: float,
    half_widths: NDArray,
    colors: ShipColors,
    rng: random.Random,
) -> None:
    """Append one superstructure <rect>, clipped to hull width."""
    x0 = rng.uniform(*spec.x0)
    x1 = rng.uniform(*spec.x1)
    if x0 >= x1:
        return
    w_frac = rng.uniform(*spec.w)

    n = len(half_widths)
    mid_idx = min(int((x0 + x1) / 2 * (n - 1)), n - 1)
    hull_hw = float(half_widths[mid_idx])

    block_hw = w_frac * 0.5
    cx = 0.5 + spec.y_off
    el = max(cx - block_hw, 0.5 - hull_hw + 0.02)
    er = min(cx + block_hw, 0.5 + hull_hw - 0.02)
    if el >= er:
        return

    y0 = x0 * lb_ratio
    y1 = x1 * lb_ratio
    fill = colors.struct_css(spec.brightness_off, rng)
    stroke = colors.struct_css(spec.brightness_off - 25, rng)
    sw = min(er - el, y1 - y0) * 0.06
    out.write(
        f'  <rect x="{_f(el)}" y="{_f(y0)}" '
        f'width="{_f(er - el)}" height="{_f(y1 - y0)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{_f(sw)}"/>\n'
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
        # Dark shadow cast by nearby structure
        w = sz * 1.2
        h = sz * 0.6
        out.write(
            f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy)}" '
            f'width="{_f(w)}" height="{_f(h)}" '
            f'fill="rgba(0,0,0,0.18)" rx="{_f(h * 0.15)}"/>\n'
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


def _write_deck_scatter_svg(
    out: StringIO,
    lb_ratio: float,
    half_widths: NDArray,
    colors: ShipColors,
    rng: random.Random,
    density: float = 3.0,
) -> None:
    """甲板上にランダムな小図形を散布してテクスチャを付加する。

    hull の <clipPath> 内部に包んだ <g> から呼び出すこと。

    Parameters
    ----------
    density
        散布密度。1 あたりの図形数 = density * lb_ratio。
        0 で散布なし。標準は 3.0。
    """
    if density <= 0:
        return

    n = len(half_widths)
    base = density * lb_ratio
    n_shapes = rng.randint(max(1, int(base * 0.6)), max(1, int(base * 1.4)) + 1)

    for _ in range(n_shapes):
        t = rng.uniform(0.04, 0.96)
        idx = min(int(t * (n - 1)), n - 1)
        hull_hw = float(half_widths[idx])
        # Skip very narrow sections (bow/stern tips)
        if hull_hw < 0.05:
            continue

        cy = t * lb_ratio
        # Keep cx strictly inside hull to avoid relying solely on clipPath
        margin = min(0.04, hull_hw * 0.15)
        half_range = hull_hw - margin
        cx = 0.5 + rng.uniform(-half_range, half_range)

        # Shape size: small relative to beam, always fits inside hull
        max_sz = min(0.06, hull_hw * 0.55)
        sz = rng.uniform(max_sz * 0.3, max_sz)

        # Colour: hull-toned with a noticeable but harmonious offset.
        # Pick a signed shift so the shape is always distinguishable from hull.
        sign = 1 if rng.random() < 0.5 else -1
        offset = sign * rng.randint(18, 38)
        # ~10% chance: accent colour (equipment / containers) — stays within
        # a muted range to blend with the overall palette.
        if rng.random() < 0.10:
            hr, hg, hb = colors.hull
            # Desaturate accent toward hull lightness to avoid eye-jarring pops
            luma = int(hr * 0.3 + hg * 0.59 + hb * 0.11)
            accent_shift = rng.randint(25, 55)
            av = max(0, min(255, luma + accent_shift))
            fill = f"rgb({av},{max(0, min(255, av - rng.randint(0, 20)))},{max(0, min(255, av - rng.randint(15, 45)))})"
            stroke_color = colors.detail_css(offset - 10)
        else:
            fill = colors.detail_css(offset)
            stroke_color = colors.detail_css(offset - sign * 12)

        sw = sz * 0.12  # unified stroke width

        kind = rng.choices(
            ["rect", "circle", "line_h", "line_v"],
            weights=[6, 2, 1, 1],
        )[0]

        if kind == "rect":
            w = sz * rng.uniform(0.7, 2.0)
            h = sz * rng.uniform(0.7, 2.0)
            out.write(
                f'  <rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" '
                f'width="{_f(w)}" height="{_f(h)}" '
                f'fill="{fill}" stroke="{stroke_color}" stroke-width="{_f(sw)}"/>\n'
            )
        elif kind == "circle":
            r = sz * 0.5
            out.write(
                f'  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" '
                f'fill="{fill}" stroke="{stroke_color}" stroke-width="{_f(sw)}"/>\n'
            )
        elif kind == "line_h":
            hw_line = sz * rng.uniform(0.8, 2.2)
            out.write(
                f'  <line x1="{_f(cx - hw_line)}" y1="{_f(cy)}" '
                f'x2="{_f(cx + hw_line)}" y2="{_f(cy)}" '
                f'stroke="{fill}" stroke-width="{_f(sw * 1.3)}"/>\n'
            )
        elif kind == "line_v":
            hl_line = sz * rng.uniform(0.8, 2.2)
            out.write(
                f'  <line x1="{_f(cx)}" y1="{_f(cy - hl_line)}" '
                f'x2="{_f(cx)}" y2="{_f(cy + hl_line)}" '
                f'stroke="{fill}" stroke-width="{_f(sw * 1.3)}"/>\n'
            )


# ── Public API ───────────────────────────────────────────────────────────


def get_ship_classes() -> list[str]:
    """Return sorted list of available ship class names."""
    return sorted(SHIP_CLASSES)


def generate_ship_svg(
    ship_class: str,
    *,
    rng: random.Random | None = None,
    hull_noise: float = 0.005,
    n_hull_points: int = 64,
    deck_scatter_density: float = 3.0,
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
    colors = sample_colors(cls.color_family, rng)

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
        f'data-lb-ratio="{_f(lb_ratio)}">\n'
    )

    # ClipPath — used to constrain deck scatter to hull outline
    out.write(
        f'  <defs>\n'
        f'    <clipPath id="h">\n'
        f'      <polygon points="{_polygon_attr(hull_pts)}"/>\n'
        f'    </clipPath>\n'
        f'  </defs>\n'
    )

    # 1) Hull polygon
    hull_stroke = colors.detail_css(-20)
    hull_sw = 0.5 / lb_ratio * 0.4
    out.write(
        f'  <polygon points="{_polygon_attr(hull_pts)}" '
        f'fill="{colors.hull_css()}" '
        f'stroke="{hull_stroke}" stroke-width="{_f(hull_sw)}" '
        f'stroke-linejoin="round"/>\n'
    )

    # 2) Superstructures
    for s in cls.structs:
        if rng.random() < s.prob:
            _write_struct_svg(out, s, lb_ratio, half_widths, colors, rng)

    # 3) Deck scatter — random small shapes clipped to hull for visual texture
    out.write('  <g clip-path="url(#h)">\n')
    _write_deck_scatter_svg(out, lb_ratio, half_widths, colors, rng, deck_scatter_density)
    out.write('  </g>\n')

    # 4) Details
    for d in cls.details:
        if rng.random() < d.prob:
            _write_detail_svg(out, d, lb_ratio, colors, rng)

    out.write("</svg>\n")
    return out.getvalue()


def generate_ships(
    output_dir: Path,
    count: int,
    *,
    types: dict[str, float] | None = None,
    seed: int | None = None,
    hull_noise: float = 0.005,
    n_hull_points: int = 64,
    deck_scatter_density: float = 3.0,
) -> None:
    """Generate synthetic ship SVG files.

    Parameters
    ----------
    output_dir
        Destination directory (created if absent).
    count
        Number of SVG files to generate.
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

    counters: dict[str, int] = {}

    for i in range(count):
        ship_class = rng.choices(classes, weights=weights, k=1)[0]
        svg = generate_ship_svg(
            ship_class, rng=rng, hull_noise=hull_noise,
            n_hull_points=n_hull_points,
            deck_scatter_density=deck_scatter_density,
        )

        counters[ship_class] = counters.get(ship_class, 0) + 1
        filename = f"{ship_class}_{counters[ship_class]:05d}.svg"
        (output_dir / filename).write_text(svg, encoding="utf-8")

        if (i + 1) % 100 == 0 or (i + 1) == count:
            logger.info("Generated %d / %d SVGs", i + 1, count)

    logger.info("Ship counts: %s", dict(sorted(counters.items())))
