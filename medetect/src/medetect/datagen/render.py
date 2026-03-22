"""SVG ship rasterisation to RGBA numpy arrays.

Parses the simple SVG elements produced by :mod:`medetect.shipgen`
(polygon, rect, circle, line) and renders them with PIL.  No external
SVG rendering library is required.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw


def parse_color(css: str) -> tuple[int, int, int]:
    """Parse ``rgb(r,g,b)`` CSS colour string.

    Returns a fallback gray for unrecognised formats.
    """
    m = re.match(r"rgb\((\d+),(\d+),(\d+)\)", css)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (128, 128, 128)


def parse_svg_metadata(svg_text: str) -> tuple[str, float]:
    """Extract ship class and L/B ratio from SVG attributes.

    Returns
    -------
    tuple[str, float]
        ``(ship_class, lb_ratio)``.  Defaults to ``("unknown", vb_h)``
        when attributes are absent.
    """
    root = ET.fromstring(svg_text)
    ship_class = root.get("data-ship-class", "unknown")
    lb_str = root.get("data-lb-ratio")
    if lb_str is not None:
        lb_ratio = float(lb_str)
    else:
        vb = root.get("viewBox", "0 0 1 1").split()
        lb_ratio = float(vb[3])
    return ship_class, lb_ratio


# ── SVG → raster ─────────────────────────────────────────────────────────

def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    el: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
) -> None:
    points_str = el.get("points", "")
    fill = el.get("fill", "rgb(128,128,128)")
    points: list[tuple[float, float]] = []
    for pair in points_str.split():
        parts = pair.split(",")
        if len(parts) == 2:
            px = (float(parts[0]) - vb_x) * sx
            py = (float(parts[1]) - vb_y) * sy
            points.append((px, py))
    if len(points) >= 3:
        r, g, b = parse_color(fill)
        draw.polygon(points, fill=(r, g, b, 255))


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    el: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
) -> None:
    x = (float(el.get("x", "0")) - vb_x) * sx
    y = (float(el.get("y", "0")) - vb_y) * sy
    w = float(el.get("width", "0")) * sx
    h = float(el.get("height", "0")) * sy
    fill = el.get("fill", "")
    stroke = el.get("stroke", "")
    if fill and fill != "none":
        r, g, b = parse_color(fill)
        draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, 255))
    if stroke and stroke != "none":
        r, g, b = parse_color(stroke)
        sw = max(1, int(float(el.get("stroke-width", "1")) * min(sx, sy)))
        draw.rectangle([x, y, x + w, y + h], outline=(r, g, b, 255), width=sw)


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    el: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
) -> None:
    cx = (float(el.get("cx", "0")) - vb_x) * sx
    cy = (float(el.get("cy", "0")) - vb_y) * sy
    r_val = float(el.get("r", "0")) * min(sx, sy)
    fill = el.get("fill", "")
    stroke = el.get("stroke", "")
    if fill and fill != "none":
        r, g, b = parse_color(fill)
        draw.ellipse(
            [cx - r_val, cy - r_val, cx + r_val, cy + r_val],
            fill=(r, g, b, 255),
        )
    if stroke and stroke != "none":
        r, g, b = parse_color(stroke)
        sw = max(1, int(float(el.get("stroke-width", "1")) * min(sx, sy)))
        draw.ellipse(
            [cx - r_val, cy - r_val, cx + r_val, cy + r_val],
            outline=(r, g, b, 255),
            width=sw,
        )


def _draw_line(
    draw: ImageDraw.ImageDraw,
    el: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
) -> None:
    x1 = (float(el.get("x1", "0")) - vb_x) * sx
    y1 = (float(el.get("y1", "0")) - vb_y) * sy
    x2 = (float(el.get("x2", "0")) - vb_x) * sx
    y2 = (float(el.get("y2", "0")) - vb_y) * sy
    stroke = el.get("stroke", "rgb(128,128,128)")
    sw = max(1, int(float(el.get("stroke-width", "1")) * min(sx, sy)))
    r, g, b = parse_color(stroke)
    draw.line([(x1, y1), (x2, y2)], fill=(r, g, b, 255), width=sw)


_DRAWER = {
    "polygon": _draw_polygon,
    "rect": _draw_rect,
    "circle": _draw_circle,
    "line": _draw_line,
}


def rasterize_ship_svg(
    svg_text: str,
    width_px: int,
    height_px: int,
    *,
    supersample: int = 4,
) -> NDArray[np.uint8]:
    """Render an SVG ship to an RGBA numpy array.

    Only handles the element types produced by :mod:`medetect.shipgen`:
    ``<polygon>``, ``<rect>``, ``<circle>``, ``<line>``.

    Internally renders at *supersample*\u00d7 resolution and downscales with
    Lanczos filtering to produce smooth, anti-aliased edges.

    Parameters
    ----------
    svg_text
        Well-formed SVG document string.
    width_px
        Output width in pixels (beam direction).
    height_px
        Output height in pixels (length direction).
    supersample
        Internal rendering scale factor (default 4).

    Returns
    -------
    NDArray[np.uint8]
        RGBA array of shape ``(height_px, width_px, 4)``.
    """
    root = ET.fromstring(svg_text)
    vb = root.get("viewBox", "0 0 1 1").split()
    vb_x, vb_y = float(vb[0]), float(vb[1])
    vb_w, vb_h = float(vb[2]), float(vb[3])

    ss = max(1, supersample)
    render_w = width_px * ss
    render_h = height_px * ss

    img = Image.new("RGBA", (render_w, render_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    sx = render_w / vb_w
    sy = render_h / vb_h

    for el in root:
        tag = el.tag.split("}")[-1]  # strip XML namespace
        drawer = _DRAWER.get(tag)
        if drawer is not None:
            drawer(draw, el, sx, sy, vb_x, vb_y)

    if ss > 1:
        img = img.resize((width_px, height_px), Image.LANCZOS)

    return np.array(img)
