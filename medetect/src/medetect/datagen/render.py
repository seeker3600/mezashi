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


def parse_color(css: str) -> tuple[int, int, int, int]:
    """Parse ``rgb(r,g,b)`` or ``rgba(r,g,b,a)`` CSS colour string.

    Returns ``(r, g, b, a)`` where *a* is 0–255.
    Falls back to opaque gray for unrecognised formats.
    """
    m = re.match(r"rgba?\((\d+),(\d+),(\d+)(?:,([\d.]+))?\)", css)
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        a = round(float(m.group(4)) * 255) if m.group(4) is not None else 255
        return r, g, b, a
    return (128, 128, 128, 255)


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

def _rotate_point(
    x: float,
    y: float,
    cx: float,
    cy: float,
    cos_a: float,
    sin_a: float,
) -> tuple[float, float]:
    """Rotate point (x, y) around centre (cx, cy)."""
    dx = x - cx
    dy = y - cy
    return (
        cx + dx * cos_a - dy * sin_a,
        cy + dx * sin_a + dy * cos_a,
    )


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    el: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
    cos_a: float = 1.0,
    sin_a: float = 0.0,
    cx_center: float = 0.0,
    cy_center: float = 0.0,
) -> None:
    points_str = el.get("points", "")
    fill = el.get("fill", "rgb(128,128,128)")
    points: list[tuple[float, float]] = []
    for pair in points_str.split():
        parts = pair.split(",")
        if len(parts) == 2:
            px = (float(parts[0]) - vb_x) * sx
            py = (float(parts[1]) - vb_y) * sy
            # Apply rotation
            px, py = _rotate_point(px, py, cx_center, cy_center, cos_a, sin_a)
            points.append((px, py))
    if len(points) >= 3:
        r, g, b, a = parse_color(fill)
        draw.polygon(points, fill=(r, g, b, a))


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    el: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
    cos_a: float = 1.0,
    sin_a: float = 0.0,
    cx_center: float = 0.0,
    cy_center: float = 0.0,
) -> None:
    x = (float(el.get("x", "0")) - vb_x) * sx
    y = (float(el.get("y", "0")) - vb_y) * sy
    w = float(el.get("width", "0")) * sx
    h = float(el.get("height", "0")) * sy
    # Build rotated rectangle corners
    corners_local = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    corners = [
        _rotate_point(px, py, cx_center, cy_center, cos_a, sin_a)
        for px, py in corners_local
    ]
    fill = el.get("fill", "")
    stroke = el.get("stroke", "")
    if fill and fill != "none":
        r, g, b, a = parse_color(fill)
        draw.polygon(corners, fill=(r, g, b, a))
    if stroke and stroke != "none":
        r, g, b, a = parse_color(stroke)
        sw = max(1, int(float(el.get("stroke-width", "1")) * min(sx, sy)))
        draw.polygon(corners, outline=(r, g, b, a), width=sw)


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    el: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
    cos_a: float = 1.0,
    sin_a: float = 0.0,
    cx_center: float = 0.0,
    cy_center: float = 0.0,
) -> None:
    cx = (float(el.get("cx", "0")) - vb_x) * sx
    cy = (float(el.get("cy", "0")) - vb_y) * sy
    r_val = float(el.get("r", "0")) * min(sx, sy)
    # Rotate circle center
    cx, cy = _rotate_point(cx, cy, cx_center, cy_center, cos_a, sin_a)
    fill = el.get("fill", "")
    stroke = el.get("stroke", "")
    if fill and fill != "none":
        r, g, b, a = parse_color(fill)
        draw.ellipse(
            [cx - r_val, cy - r_val, cx + r_val, cy + r_val],
            fill=(r, g, b, a),
        )
    if stroke and stroke != "none":
        r, g, b, a = parse_color(stroke)
        sw = max(1, int(float(el.get("stroke-width", "1")) * min(sx, sy)))
        draw.ellipse(
            [cx - r_val, cy - r_val, cx + r_val, cy + r_val],
            outline=(r, g, b, a),
            width=sw,
        )


def _draw_line(
    draw: ImageDraw.ImageDraw,
    el: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
    cos_a: float = 1.0,
    sin_a: float = 0.0,
    cx_center: float = 0.0,
    cy_center: float = 0.0,
) -> None:
    x1 = (float(el.get("x1", "0")) - vb_x) * sx
    y1 = (float(el.get("y1", "0")) - vb_y) * sy
    x2 = (float(el.get("x2", "0")) - vb_x) * sx
    y2 = (float(el.get("y2", "0")) - vb_y) * sy
    # Rotate line endpoints
    x1, y1 = _rotate_point(x1, y1, cx_center, cy_center, cos_a, sin_a)
    x2, y2 = _rotate_point(x2, y2, cx_center, cy_center, cos_a, sin_a)
    stroke = el.get("stroke", "rgb(128,128,128)")
    sw = max(1, int(float(el.get("stroke-width", "1")) * min(sx, sy)))
    r, g, b, a = parse_color(stroke)
    draw.line([(x1, y1), (x2, y2)], fill=(r, g, b, a), width=sw)


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
    angle_deg: float = 0.0,
    supersample: int = 4,
) -> NDArray[np.uint8]:
    """Render an SVG ship to an RGBA numpy array, optionally rotated.

    Only handles the element types produced by :mod:`medetect.shipgen`:
    ``<polygon>``, ``<rect>``, ``<circle>``, ``<line>``.

    Rotation is applied to the vector geometry before rasterization,
    producing higher-quality edges than post-hoc bitmap rotation.

    Internally renders at *supersample*\u00d7 resolution and downscales with
    Lanczos filtering to produce smooth, anti-aliased edges.

    Parameters
    ----------
    svg_text
        Well-formed SVG document string.
    width_px
        Ship width in pixels (beam direction, before rotation).
    height_px
        Ship height in pixels (length direction, before rotation).
    angle_deg
        Rotation angle in degrees (default 0.0).
    supersample
        Internal rendering scale factor (default 4).

    Returns
    -------
    NDArray[np.uint8]
        RGBA array.  When *angle_deg* is 0 the shape is
        ``(height_px, width_px, 4)``.  Otherwise it is the tight
        rotated bounding-box: ``(out_h, out_w, 4)``.
    """
    import math

    root = ET.fromstring(svg_text)
    vb = root.get("viewBox", "0 0 1 1").split()
    vb_x, vb_y = float(vb[0]), float(vb[1])
    vb_w, vb_h = float(vb[2]), float(vb[3])

    ss = max(1, supersample)
    render_w = width_px * ss
    render_h = height_px * ss

    # Compute rotation parameters
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    abs_cos = abs(cos_a)
    abs_sin = abs(sin_a)

    # For non-zero angles, allocate a larger canvas to hold the rotated content
    if angle_deg != 0.0:
        rotated_w = int(render_w * abs_cos + render_h * abs_sin) + 2
        rotated_h = int(render_w * abs_sin + render_h * abs_cos) + 2
    else:
        rotated_w = render_w
        rotated_h = render_h

    img = Image.new("RGBA", (rotated_w, rotated_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    sx = render_w / vb_w
    sy = render_h / vb_h

    # Shift the viewBox origin so that SVG content is centred on the
    # expanded canvas.  This makes the content centre coincide with
    # the canvas centre, so rotating around the canvas centre is correct.
    off_x = (rotated_w - render_w) / 2.0
    off_y = (rotated_h - render_h) / 2.0
    adj_vb_x = vb_x - off_x / sx
    adj_vb_y = vb_y - off_y / sy

    # Rotation centre = canvas centre = content centre after offset
    cx_center = rotated_w / 2.0
    cy_center = rotated_h / 2.0

    for el in root:
        tag = el.tag.split("}")[-1]  # strip XML namespace
        drawer = _DRAWER.get(tag)
        if drawer is not None:
            drawer(draw, el, sx, sy, adj_vb_x, adj_vb_y, cos_a, sin_a, cx_center, cy_center)

    # Make ship pixels fully opaque.
    # The hull polygon is the first polygon in the SVG and defines the ship
    # interior.  Re-rendering it to a separate mask and applying that mask
    # as the alpha channel achieves two things:
    #   1. Shadow/overlay elements that extend past the hull edge are clipped.
    #   2. If Pillow's ImageDraw writes rgba alpha values directly (older
    #      versions), hull pixels are restored to alpha=255.
    # Overall ship transparency is controlled externally by blend_ship().
    hull_polygon_el: ET.Element | None = None
    for el in root:
        if el.tag.split("}")[-1] == "polygon":
            hull_polygon_el = el
            break
    if hull_polygon_el is not None:
        hull_mask = Image.new("RGBA", img.size, (0, 0, 0, 0))
        _draw_polygon(
            ImageDraw.Draw(hull_mask),
            hull_polygon_el, sx, sy, adj_vb_x, adj_vb_y,
            cos_a, sin_a, cx_center, cy_center,
        )
        arr = np.array(img)
        arr[:, :, 3] = np.array(hull_mask)[:, :, 3]
        img = Image.fromarray(arr)

    # Output size = rotated bounding-box of (width_px, height_px)
    if angle_deg != 0.0:
        out_w = max(1, round(width_px * abs_cos + height_px * abs_sin))
        out_h = max(1, round(width_px * abs_sin + height_px * abs_cos))
    else:
        out_w = width_px
        out_h = height_px

    # Downscale from supersample canvas to final output size
    if rotated_w != out_w or rotated_h != out_h:
        img = img.resize((out_w, out_h), Image.LANCZOS)

    return np.array(img)
