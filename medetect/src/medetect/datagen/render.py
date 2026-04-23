"""SVG ship rasterisation to RGBA numpy arrays.

Parses the simple SVG elements produced by :mod:`medetect.shipgen`
(polygon, rect, circle, line) and renders them with PIL.  No external
SVG rendering library is required.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter

from medetect.datagen.svg import parse_svg_metadata


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


def _strip_tag(tag: str) -> str:
    return tag.split("}")[-1]


def _parse_points(points_str: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for pair in points_str.split():
        parts = pair.split(",")
        if len(parts) != 2:
            continue
        points.append((float(parts[0]), float(parts[1])))
    return points


def _find_hull_elements(
    root: ET.Element,
) -> tuple[list[tuple[float, float]], ET.Element | None]:
    """Return hull polygon points and the drawable hull polygon element if present."""
    hull_points: list[tuple[float, float]] = []
    for el in root.iter():
        if _strip_tag(el.tag) != "clipPath":
            continue
        for child in el:
            if _strip_tag(child.tag) == "polygon":
                hull_points = _parse_points(child.get("points", ""))
                if hull_points:
                    break
        if hull_points:
            break

    drawable_hull: ET.Element | None = None
    hull_signature = tuple(hull_points)
    for child in root:
        if _strip_tag(child.tag) != "polygon":
            continue
        child_points = tuple(_parse_points(child.get("points", "")))
        if hull_signature and child_points == hull_signature:
            drawable_hull = child
            break
        if not hull_signature and child.get("fill") not in {None, "", "none"}:
            hull_points = list(child_points)
            drawable_hull = child
            break

    return hull_points, drawable_hull


def extract_hull_polygon(svg_text: str) -> list[tuple[float, float]]:
    """Extract the ship hull polygon in SVG viewBox coordinates."""
    root = ET.fromstring(svg_text)
    hull_points, _drawable_hull = _find_hull_elements(root)
    if not hull_points:
        raise ValueError("Hull polygon not found in SVG")
    return hull_points


def extract_hull_fill(svg_text: str) -> tuple[int, int, int, int]:
    """Extract the RGBA fill colour of the drawable hull polygon."""
    root = ET.fromstring(svg_text)
    _hull_points, drawable_hull = _find_hull_elements(root)
    if drawable_hull is None:
        return (128, 128, 128, 255)
    return parse_color(drawable_hull.get("fill", "rgb(128,128,128)"))


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


def _composite_over(
    img: Image.Image,
    draw_fn: "Callable[[ImageDraw.ImageDraw], None]",
) -> None:
    """Porter-Duff 'over' compositing: draw to a temp layer and composite onto img."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    img.alpha_composite(layer)


def _draw_polygon(
    img: Image.Image,
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
        if a == 255:
            ImageDraw.Draw(img).polygon(points, fill=(r, g, b, a))
        else:
            _composite_over(img, lambda d: d.polygon(points, fill=(r, g, b, a)))


def _draw_rect(
    img: Image.Image,
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
        if a == 255:
            ImageDraw.Draw(img).polygon(corners, fill=(r, g, b, a))
        else:
            _composite_over(img, lambda d: d.polygon(corners, fill=(r, g, b, a)))
    if stroke and stroke != "none":
        r, g, b, a = parse_color(stroke)
        sw = max(1, int(float(el.get("stroke-width", "1")) * min(sx, sy)))
        if a == 255:
            ImageDraw.Draw(img).polygon(corners, outline=(r, g, b, a), width=sw)
        else:
            _composite_over(img, lambda d: d.polygon(corners, outline=(r, g, b, a), width=sw))


def _draw_circle(
    img: Image.Image,
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
    bbox = [cx - r_val, cy - r_val, cx + r_val, cy + r_val]
    fill = el.get("fill", "")
    stroke = el.get("stroke", "")
    if fill and fill != "none":
        r, g, b, a = parse_color(fill)
        if a == 255:
            ImageDraw.Draw(img).ellipse(bbox, fill=(r, g, b, a))
        else:
            _composite_over(img, lambda d: d.ellipse(bbox, fill=(r, g, b, a)))
    if stroke and stroke != "none":
        r, g, b, a = parse_color(stroke)
        sw = max(1, int(float(el.get("stroke-width", "1")) * min(sx, sy)))
        if a == 255:
            ImageDraw.Draw(img).ellipse(bbox, outline=(r, g, b, a), width=sw)
        else:
            _composite_over(img, lambda d: d.ellipse(bbox, outline=(r, g, b, a), width=sw))


def _draw_line(
    img: Image.Image,
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
    pts = [(x1, y1), (x2, y2)]
    if a == 255:
        ImageDraw.Draw(img).line(pts, fill=(r, g, b, a), width=sw)
    else:
        _composite_over(img, lambda d: d.line(pts, fill=(r, g, b, a), width=sw))


_DRAWER = {
    "polygon": _draw_polygon,
    "rect": _draw_rect,
    "circle": _draw_circle,
    "line": _draw_line,
}


def _resize_premultiplied(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Resize an RGBA image using premultiplied alpha to avoid dark-fringe artefacts.

    PIL's Lanczos filter treats each channel independently (straight alpha).
    When the canvas background is transparent black ``(0,0,0,0)``, interpolating
    ship-edge pixels with those transparent-black neighbours pulls RGB toward
    black even at moderate alpha values.  Premultiplying RGB by alpha before
    downscaling and reversing afterwards (unpremultiply) prevents this colour
    bleed.
    """
    arr = np.array(img, dtype=np.float32)        # H × W × 4
    alpha = arr[:, :, 3:4] / 255.0              # 0-1, broadcast-ready
    arr[:, :, :3] *= alpha                       # premultiply RGB
    premul = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    premul = premul.resize((out_w, out_h), Image.LANCZOS)
    arr2 = np.array(premul, dtype=np.float32)
    alpha2 = arr2[:, :, 3:4] / 255.0
    # Unpremultiply — guard against division by near-zero alpha
    safe_alpha = np.where(alpha2 > 1e-3, alpha2, 1.0)
    arr2[:, :, :3] /= safe_alpha
    arr2[:, :, :3] = np.clip(arr2[:, :, :3], 0, 255)
    return Image.fromarray(arr2.astype(np.uint8), "RGBA")


def resize_rgba_premultiplied(
    rgba: NDArray[np.uint8],
    out_w: int,
    out_h: int,
) -> NDArray[np.uint8]:
    """Resize an RGBA array with premultiplied-alpha filtering."""
    img = Image.fromarray(rgba, "RGBA")
    resized = _resize_premultiplied(img, out_w, out_h)
    return np.array(resized, dtype=np.uint8)


def gaussian_blur_rgba_premultiplied(
    rgba: NDArray[np.uint8],
    radius: float,
) -> NDArray[np.uint8]:
    """Apply Gaussian blur to RGBA without darkening semi-transparent edges."""
    if radius <= 0.0:
        return np.array(rgba, copy=True)

    arr = rgba.astype(np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    arr[:, :, :3] *= alpha

    premul = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    premul = premul.filter(ImageFilter.GaussianBlur(radius=radius))

    blurred = np.array(premul, dtype=np.float32)
    alpha2 = blurred[:, :, 3:4] / 255.0
    safe_alpha = np.where(alpha2 > 1e-3, alpha2, 1.0)
    blurred[:, :, :3] /= safe_alpha
    blurred[:, :, :3] = np.where(alpha2 > 1e-3, blurred[:, :, :3], 0.0)
    blurred[:, :, :3] = np.clip(blurred[:, :, :3], 0, 255)

    return blurred.astype(np.uint8)


def _draw_elements(
    img: Image.Image,
    parent: ET.Element,
    sx: float,
    sy: float,
    vb_x: float,
    vb_y: float,
    cos_a: float,
    sin_a: float,
    cx_center: float,
    cy_center: float,
    skip_element: ET.Element | None = None,
) -> None:
    """Recursively render child elements, descending into ``<g>`` groups."""
    for el in parent:
        if el is skip_element:
            continue
        tag = _strip_tag(el.tag)
        if tag == "g":
            _draw_elements(
                img,
                el,
                sx,
                sy,
                vb_x,
                vb_y,
                cos_a,
                sin_a,
                cx_center,
                cy_center,
                skip_element,
            )
        else:
            drawer = _DRAWER.get(tag)
            if drawer is not None:
                drawer(img, el, sx, sy, vb_x, vb_y, cos_a, sin_a, cx_center, cy_center)


def rasterize_ship_svg(
    svg_text: str,
    width_px: int,
    height_px: int,
    *,
    angle_deg: float = 0.0,
    supersample: int = 4,
    exclude_hull: bool = False,
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
    exclude_hull
        When ``True``, skip the top-level hull fill/stroke polygon and render
        only the clipped shading/details layers.

    Returns
    -------
    NDArray[np.uint8]
        RGBA array.  When *angle_deg* is 0 the shape is
        ``(height_px, width_px, 4)``.  Otherwise it is the tight
        rotated bounding-box: ``(out_h, out_w, 4)``.
    """
    import math

    root = ET.fromstring(svg_text)
    _hull_points, drawable_hull = _find_hull_elements(root)
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

    _draw_elements(
        img,
        root,
        sx,
        sy,
        adj_vb_x,
        adj_vb_y,
        cos_a,
        sin_a,
        cx_center,
        cy_center,
        drawable_hull if exclude_hull else None,
    )

    # Output size = rotated bounding-box of (width_px, height_px)
    if angle_deg != 0.0:
        out_w = max(1, round(width_px * abs_cos + height_px * abs_sin))
        out_h = max(1, round(width_px * abs_sin + height_px * abs_cos))
    else:
        out_w = width_px
        out_h = height_px

    # Downscale from supersample canvas to final output size.
    # Use premultiplied-alpha resize to avoid dark fringing at ship edges.
    if rotated_w != out_w or rotated_h != out_h:
        img = _resize_premultiplied(img, out_w, out_h)

    return np.array(img)
