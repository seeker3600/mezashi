"""Synthetic ship detection dataset generator.

Composites SVG ship silhouettes onto Sentinel-2 background imagery to
produce a YOLO OBB (oriented bounding box) training dataset.  Water
regions are identified via the Sentinel-2 L2A Scene Classification Layer
(SCL band, value 6 = water).

Output structure::

    output_dir/
        images/train/  ← PNG tiles
        labels/train/  ← YOLO OBB label files
        dataset.yaml   ← YOLO config
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rasterio
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter
from rasterio.windows import Window
from tqdm import tqdm

from medetect.datagen.render import parse_svg_metadata, rasterize_ship_svg
from medetect.datagen.wake import MotionState, pick_motion_state, render_wake
from medetect.datagen.water_mask import (
    CoastlineIndex,
    erode_mask,
    make_water_mask_from_coastline,
    make_water_mask_from_rgb,
    make_water_mask_from_scl,
)

logger = logging.getLogger(__name__)


# ── Real-world ship lengths (metres) ─────────────────────────────────────

SHIP_LENGTHS_M: dict[str, tuple[float, float]] = {
    "patrol": (30.0, 80.0),
    "corvette": (80.0, 110.0),
    "frigate": (110.0, 150.0),
    "destroyer": (150.0, 190.0),
    "destroyer_stealth": (140.0, 180.0),
    "carrier": (260.0, 340.0),
    "amphib_assault": (200.0, 260.0),
    "lst_lpd": (120.0, 200.0),
    "supply": (150.0, 210.0),
    "fishing_squid_jigger": (20.0, 50.0),
    "fishing_trawler": (15.0, 40.0),
    "fishing_purse_seiner": (25.0, 60.0),
    "fishing_longliner": (20.0, 45.0),
}

_DEFAULT_LENGTH_M = (30.0, 100.0)

# Mean pixel value (0-255) below which a tile is considered a satellite blackout area.
_DARK_TILE_THRESHOLD: float = 10.0


# ── Pure helpers (easily testable) ────────────────────────────────────────


def augment_tile(
    tile: NDArray[np.uint8],
    rng: random.Random,
) -> NDArray[np.uint8]:
    """Apply random colour augmentation to a background tile.

    Applies per-channel gain, gamma correction, and brightness offset to
    introduce visual diversity among tiles that may originate from the same
    ocean region.  The augmentation is intentionally moderate so that the
    result still looks like a plausible satellite image.
    """
    img = tile.astype(np.float32)

    # Per-channel multiplicative gain  (±15 %)
    for c in range(3):
        img[:, :, c] *= rng.uniform(0.85, 1.15)

    # Global gamma  (0.8 – 1.2)
    gamma = rng.uniform(0.8, 1.2)
    img = np.clip(img, 0, 255)
    img = 255.0 * (img / 255.0) ** gamma

    # Brightness offset  (±10)
    img += rng.uniform(-10, 10)

    return np.clip(img, 0, 255).astype(np.uint8)


def is_dark_tile(tile: NDArray[np.uint8], threshold: float = _DARK_TILE_THRESHOLD) -> bool:
    """Return True when the tile is a satellite blackout / no-data area.

    Sentinel-2 images can contain strip-shaped black regions caused by
    detector gaps or missing acquisitions.  A mean pixel value below
    *threshold* reliably identifies such tiles.
    """
    return float(tile.mean()) < threshold


def make_nodata_mask(tile: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Return ``True`` for pixels that are pure black (#000000) — no-data areas.

    Satellite products such as Sentinel-2 use RGB (0, 0, 0) to mark pixels
    outside the swath or that were masked during processing.  These must be
    excluded from water masking and ship placement.
    """
    return (tile[:, :, 0] == 0) & (tile[:, :, 1] == 0) & (tile[:, :, 2] == 0)


def compute_ship_pixel_size(
    ship_class: str,
    lb_ratio: float,
    resolution_m: float,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
) -> tuple[int, int]:
    """Compute ship raster size ``(beam_px, length_px)`` for the tile resolution.

    A random length within the real-world range for *ship_class* is chosen,
    then converted to pixels at the given *resolution_m*.

    Parameters
    ----------
    length_range
        Global ``(min_m, max_m)`` clamp applied on top of the per-class range.
        When *None*, only the per-class range is used.
    length_exponent
        Controls the size-frequency distribution.  ``1.0`` = log-uniform
        (default, equal probability per multiplicative factor).  ``> 1.0``
        produces more small ships; ``< 1.0`` (towards 0) gives a more
        uniform distribution.
    """
    lo, hi = SHIP_LENGTHS_M.get(ship_class, _DEFAULT_LENGTH_M)
    if length_range is not None:
        lo = max(lo, length_range[0])
        hi = min(hi, length_range[1])
        if lo > hi:
            lo, hi = length_range[0], length_range[1]
    # Generalized log-power distribution:
    #   exponent=1.0 → log-uniform (equal probability per multiplicative factor)
    #   exponent>1.0 → more small ships
    #   exponent<1.0 → more uniform (less small-biased)
    lo = max(lo, 1.0)  # guard against log(0)
    u = rng.random()
    t = u ** length_exponent
    length_m = lo * (hi / lo) ** t
    beam_m = length_m / lb_ratio

    length_px = max(3, round(length_m / resolution_m))
    beam_px = max(2, round(beam_m / resolution_m))
    return beam_px, length_px


def compute_obb_corners(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_rad: float,
) -> list[tuple[float, float]]:
    """Compute oriented bounding-box corners.

    Returns four ``(x, y)`` tuples in clockwise order, in pixel
    coordinates (not normalised).
    """
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


def find_water_position(
    water_mask: NDArray[np.bool_],
    ship_w: int,
    ship_h: int,
    angle_rad: float,
    rng: random.Random,
    *,
    max_attempts: int = 100,
) -> tuple[int, int] | None:
    """Find a valid centre position for a ship on the water mask.

    The rotated bounding box of the ship must lie entirely within water.
    Returns ``(cx, cy)`` or ``None`` if no position could be found.
    """
    img_h, img_w = water_mask.shape

    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    bbox_w = int(ship_w * cos_a + ship_h * sin_a) + 2
    bbox_h = int(ship_w * sin_a + ship_h * cos_a) + 2
    half_w = bbox_w // 2
    half_h = bbox_h // 2

    if 2 * half_w >= img_w or 2 * half_h >= img_h:
        return None

    for _ in range(max_attempts):
        cx = rng.randint(half_w, img_w - half_w - 1)
        cy = rng.randint(half_h, img_h - half_h - 1)
        region = water_mask[cy - half_h : cy + half_h, cx - half_w : cx + half_w]
        if region.all():
            return cx, cy

    return None


def blend_ship(
    background: NDArray[np.uint8],
    ship_rgba: NDArray[np.uint8],
    cx: int,
    cy: int,
    alpha_factor: float = 0.85,
    water_tint: NDArray[np.float32] | None = None,
) -> None:
    """Alpha-composite *ship_rgba* onto *background* centred at ``(cx, cy)``.

    Modifies *background* in place.
    """
    sh, sw = ship_rgba.shape[:2]
    x0, y0 = cx - sw // 2, cy - sh // 2

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)

    bh, bw = background.shape[:2]
    copy_w = min(sw - src_x0, bw - dst_x0)
    copy_h = min(sh - src_y0, bh - dst_y0)
    if copy_w <= 0 or copy_h <= 0:
        return

    ship_crop = ship_rgba[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    bg_crop = background[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w]

    alpha = (ship_crop[:, :, 3:4].astype(np.float32) / 255.0) * alpha_factor
    ship_rgb = ship_crop[:, :, :3].astype(np.float32)

    # Slightly tint ship colours toward surrounding water for realism
    if water_tint is not None:
        ship_rgb = ship_rgb * 0.82 + water_tint * 0.18

    blended = bg_crop.astype(np.float32) * (1.0 - alpha) + ship_rgb * alpha
    background[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = (
        blended.clip(0, 255).astype(np.uint8)
    )


# ── SVG loading helpers ───────────────────────────────────────────────────


def _load_svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class _SvgMeta(NamedTuple):
    """Pre-read metadata for one SVG ship file."""

    path: Path
    lb_ratio: float


def _load_svg_metas(svg_files: list[Path]) -> list[_SvgMeta]:
    """Read lb_ratio from every SVG file up-front for weighted selection."""
    metas: list[_SvgMeta] = []
    for path in svg_files:
        _cls, lb = parse_svg_metadata(path.read_text(encoding="utf-8"))
        metas.append(_SvgMeta(path=path, lb_ratio=lb))
    return metas


# ── Worker-process shared state ───────────────────────────────────────────

# svg_metas and coastline_index are set once per worker process via the
# ProcessPoolExecutor initializer to avoid pickling them with every task.
_worker_svg_metas: list[_SvgMeta] | None = None
_worker_coastline_index: CoastlineIndex | None = None


def _worker_init(
    svg_dir: Path | None,
    coastline_path: Path | None = None,
) -> None:
    """Initializer for each worker process.

    Loads SVG metadata from *svg_dir* and (optionally) the coastline
    spatial index from *coastline_path* into process-local globals so
    that tasks can reference them without re-pickling on every call.
    """
    global _worker_svg_metas, _worker_coastline_index  # noqa: PLW0603
    if svg_dir is not None:
        svg_files = sorted(svg_dir.glob("*.svg"))
        _worker_svg_metas = _load_svg_metas(svg_files)
    else:
        _worker_svg_metas = None

    if coastline_path is not None:
        _worker_coastline_index = CoastlineIndex(coastline_path)
    else:
        _worker_coastline_index = None


def _natural_lb_ratio(length_m: float) -> float:
    """Empirical L/B ratio typical for a vessel of the given length.

    Derived from real-world data: smaller vessels are proportionally
    wider (lower L/B) than large ships.
    Linear approximation: lb ≈ 3.0 + 0.03 × length_m, capped at 10.
    """
    return min(10.0, 3.0 + 0.03 * length_m)


def _svg_lb_weight(lb_ratio: float, target_length_m: float) -> float:
    """Preference weight for an SVG with *lb_ratio* when generating ships
    of *target_length_m* metres.

    lb_ratios within 1.5× the natural value for that length score 1.0;
    those exceeding it are exponentially down-weighted so that unnaturally
    thin silhouettes are avoided for small ships.
    """
    natural = _natural_lb_ratio(target_length_m)
    excess = max(0.0, lb_ratio - natural * 1.5)
    return math.exp(-excess / 2.0)


def _pick_svg(
    svg_metas: list[_SvgMeta] | None,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
) -> str:
    """Return SVG text weighted by lb_ratio suitability for *length_range*.

    Smaller ships are proportionally wider (lower L/B ratio), so SVGs
    with lb_ratio close to the natural value for the target length are
    preferred.  Both pre-generated SVG files and on-the-fly shipgen ship
    classes are weighted accordingly.
    """
    target_m: float | None = (
        (length_range[0] + length_range[1]) / 2.0 if length_range is not None else None
    )

    if svg_metas:
        if target_m is not None:
            weights = [_svg_lb_weight(m.lb_ratio, target_m) for m in svg_metas]
            (meta,) = rng.choices(svg_metas, weights=weights, k=1)
        else:
            meta = rng.choice(svg_metas)
        return _load_svg(meta.path)

    from medetect.shipgen.gen import generate_ship_svg, get_ship_classes
    from medetect.shipgen.ship_class import SHIP_CLASSES

    classes = get_ship_classes()
    if target_m is not None:
        weights = [
            _svg_lb_weight(
                (SHIP_CLASSES[c].lb[0] + SHIP_CLASSES[c].lb[1]) / 2.0,
                target_m,
            )
            for c in classes
        ]
        (cls,) = rng.choices(classes, weights=weights, k=1)
    else:
        cls = rng.choice(classes)
    return generate_ship_svg(cls, rng=rng)


# ── Tile reading ──────────────────────────────────────────────────────────


def _read_tile(
    src: rasterio.DatasetReader,
    col: int,
    row: int,
    size: int,
) -> NDArray[np.uint8]:
    """Read an RGB tile from an open rasterio dataset."""
    window = Window(col, row, size, size)
    bands = min(src.count, 3)
    data = src.read(list(range(1, bands + 1)), window=window)
    # (C, H, W) → (H, W, C)
    return np.moveaxis(data, 0, -1).astype(np.uint8)


def _read_scl_tile(
    scl_path: Path,
    col: int,
    row: int,
    size: int,
    target_size: int,
) -> NDArray[np.uint8] | None:
    """Read corresponding SCL tile, resampled to *target_size*."""
    if not scl_path.exists():
        return None
    with rasterio.open(scl_path) as scl_src:
        # SCL at 20 m vs visual at 10 m → factor 2
        scale = scl_src.res[0] / 10.0  # Should be ~2.0
        scl_col = int(col / scale)
        scl_row = int(row / scale)
        scl_size = max(1, int(size / scale))

        scl_col = min(scl_col, scl_src.width - scl_size)
        scl_row = min(scl_row, scl_src.height - scl_size)
        scl_col = max(0, scl_col)
        scl_row = max(0, scl_row)

        window = Window(scl_col, scl_row, scl_size, scl_size)
        scl = scl_src.read(1, window=window)

    # Upsample to target_size via nearest-neighbour
    img = Image.fromarray(scl)
    img = img.resize((target_size, target_size), Image.NEAREST)
    return np.array(img)


def _scl_path_for(visual_path: Path) -> Path:
    """Derive SCL file path from a visual TIF path."""
    name = visual_path.name
    scl_name = name.replace("_visual.tif", "_SCL_20m.tif")
    return visual_path.parent / scl_name


# ── Ship rendering pipeline ──────────────────────────────────────────────


def _composite_rgba(
    dst: NDArray[np.uint8],
    src: NDArray[np.uint8],
    x0: int,
    y0: int,
) -> None:
    """Porter-Duff source-over composite *src* onto *dst* at ``(x0, y0)``.

    Both arrays are RGBA uint8.  *dst* is modified in place.
    """
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]

    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw = min(sw - sx0, dw - dx0)
    ch = min(sh - sy0, dh - dy0)
    if cw <= 0 or ch <= 0:
        return

    s = src[sy0 : sy0 + ch, sx0 : sx0 + cw].astype(np.float32)
    d = dst[dy0 : dy0 + ch, dx0 : dx0 + cw].astype(np.float32)

    sa = s[:, :, 3:4] / 255.0
    da = d[:, :, 3:4] / 255.0

    out_a = sa + da * (1.0 - sa)
    safe = np.where(out_a > 0, out_a, 1.0)
    out_rgb = (s[:, :, :3] * sa + d[:, :, :3] * da * (1.0 - sa)) / safe

    dst[dy0 : dy0 + ch, dx0 : dx0 + cw, :3] = out_rgb.clip(0, 255).astype(np.uint8)
    dst[dy0 : dy0 + ch, dx0 : dx0 + cw, 3:4] = (out_a * 255.0).clip(0, 255).astype(np.uint8)


def _blend_rgba_layer(
    background: NDArray[np.uint8],
    layer: NDArray[np.uint8],
    alpha_factor: float,
    water_tint: NDArray[np.float32],
) -> None:
    """Alpha-composite an RGBA *layer* onto an RGB *background* in place."""
    alpha = (layer[:, :, 3:4].astype(np.float32) / 255.0) * alpha_factor
    ship_rgb = layer[:, :, :3].astype(np.float32) * 0.82 + water_tint * 0.18
    blended = background.astype(np.float32) * (1.0 - alpha) + ship_rgb * alpha
    background[:] = blended.clip(0, 255).astype(np.uint8)


def _render_ship(
    svg_text: str,
    resolution_m: float,
    rng: random.Random,
    blur_sigma: float,
    length_range: tuple[float, float] | None = None,
    angle_deg: float = 0.0,
    length_exponent: float = 1.0,
) -> tuple[NDArray[np.uint8], str, int, int, float]:
    """Render one ship and return ``(rgba, class_name, beam_px, length_px, lb_ratio)``.

    The ship is rotated during SVG rasterization (not post-hoc) for better quality.
    """
    ship_class, lb_ratio = parse_svg_metadata(svg_text)
    beam_px, length_px = compute_ship_pixel_size(
        ship_class, lb_ratio, resolution_m, rng, length_range, length_exponent,
    )

    # Rasterize with rotation applied during SVG rendering
    rgba = rasterize_ship_svg(svg_text, beam_px, length_px, angle_deg=angle_deg)

    # Gaussian blur to simulate satellite PSF
    if blur_sigma > 0 and min(beam_px, length_px) > 2:
        img = Image.fromarray(rgba)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        rgba = np.array(img)

    return rgba, ship_class, beam_px, length_px, lb_ratio


# _rotate_ship() removed: rotation now happens during SVG rasterization in _render_ship()


def _stamp_occupancy(
    occupancy: NDArray[np.bool_],
    cx: int,
    cy: int,
    w: int,
    h: int,
    angle_rad: float,
    margin: int = 2,
) -> None:
    """回転 OBB フットプリント（+ margin px）を占有済みとしてマークする（in-place）。"""
    corners = compute_obb_corners(
        float(cx), float(cy),
        float(w + margin * 2), float(h + margin * 2),
        angle_rad,
    )
    img = Image.fromarray(occupancy.astype(np.uint8) * 255)
    draw = ImageDraw.Draw(img)
    draw.polygon(corners, fill=255)
    occupancy[:] = np.array(img) > 0


# ── Cluster logic ─────────────────────────────────────────────────────────


def _place_cluster(
    water_mask: NDArray[np.bool_],
    occupancy: NDArray[np.bool_],
    svg_metas: list[_SvgMeta] | None,
    resolution_m: float,
    rng: random.Random,
    cluster_size_range: tuple[int, int],
    blur_sigma: float,
    alpha_range: tuple[float, float],
    class_id: int,
    image_size: int,
    background: NDArray[np.uint8],
    length_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
    size_threshold: float | None = None,
    mixed_prob: float = 0.5,
) -> list[str]:
    """Place a cluster of ships side-by-side.  Returns label lines.

    Ships are laid out perpendicular to the heading direction so their
    hulls touch.  Two cluster modes are supported:

    - **Uniform** (``mixed=False``): all ships are the same visual size
      (±10 %), like a naval formation or moored fleet of sister ships.
    - **Mixed** (``mixed=True``): each ship is independently sized and
      typed, like a busy anchorage or commercial harbour.

    The mode is chosen probabilistically per cluster via *mixed_prob*.
    """
    n_ships = rng.randint(*cluster_size_range)
    base_angle = rng.uniform(0, 360)
    labels: list[str] = []

    # Decide cluster mode for this invocation.
    mixed = rng.random() < mixed_prob

    # Pick the first ship (will be rendered with angle in the loop,
    # but we need to determine size for base-position search)
    svg_text = _pick_svg(svg_metas, rng, length_range)
    # Render without rotation just to get the size
    rgba0, cls0, bw0, lh0, lb0 = _render_ship(
        svg_text, resolution_m, rng, blur_sigma, length_range,
        length_exponent=length_exponent,
    )
    angle0_rad = math.radians(base_angle)

    # Find a base position on available (water & unoccupied) area
    available = water_mask & ~occupancy
    pos = find_water_position(available, bw0 * 2, lh0 * 2, angle0_rad, rng)
    if pos is None:
        return labels

    base_cx, base_cy = pos
    # Running perpendicular cursor: starts at 0, advances by each ship's beam
    cursor = 0

    # Unified cluster alpha and water tint — ships in the same cluster
    # share environmental conditions.
    cluster_alpha = rng.uniform(*alpha_range)
    water_tint = _sample_water_tint(background, base_cx, base_cy)

    # Accumulate all cluster ships into an RGBA buffer, then blend once.
    # This avoids double-blending artefacts in the gaps between hulls.
    cluster_buf = np.zeros((image_size, image_size, 4), dtype=np.uint8)
    placed: list[tuple[int, int, int, int, float, int]] = []  # (cx, cy, bw, lh, angle_rad, cid)

    # Snapshot of occupancy BEFORE this cluster starts.
    # Each ship checks against this to avoid landing on OTHER clusters/ships,
    # while still allowing intra-cluster side-by-side proximity.
    pre_occupancy = occupancy.copy()

    for i in range(n_ships):
        # Small angle jitter within the cluster
        angle_deg = base_angle + rng.uniform(-10, 10)
        angle_rad = math.radians(angle_deg)

        if i == 0:
            # First ship: already rendered above, re-render with jitter angle.
            rotated, cls_name, bw, lh, lb = _render_ship(
                svg_text, resolution_m, rng, blur_sigma, length_range,
                angle_deg=angle_deg, length_exponent=length_exponent,
            )
        elif mixed:
            # Mixed cluster: each subsequent ship is independently typed and
            # sized — drawn from the full length_range without size pinning.
            svg_text_i = _pick_svg(svg_metas, rng, length_range)
            rotated, cls_name, bw, lh, lb = _render_ship(
                svg_text_i, resolution_m, rng, blur_sigma, length_range,
                angle_deg=angle_deg, length_exponent=length_exponent,
            )
        else:
            # Uniform cluster: different SVG but pinned to ±10% of first ship's
            # pixel size so all hulls look like they belong to the same class.
            svg_text = _pick_svg(svg_metas, rng, length_range)
            _cls, lb = parse_svg_metadata(svg_text)
            scale = rng.uniform(0.9, 1.1)
            jit_bw = max(2, round(bw0 * scale))
            jit_lh = max(3, round(lh0 * scale))
            # Render rotated SVG
            rotated = rasterize_ship_svg(
                svg_text, jit_bw, jit_lh, angle_deg=angle_deg
            )
            if blur_sigma > 0 and min(jit_bw, jit_lh) > 2:
                img = Image.fromarray(rotated)
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
                rotated = np.array(img)
            cls_name, bw, lh = cls0, jit_bw, jit_lh

        # Offset across the beam (side-by-side / hull-to-hull).
        offset = cursor + bw // 2
        cx = base_cx + int(offset * math.cos(angle_rad))
        cy = base_cy + int(offset * math.sin(angle_rad))
        cursor += bw + rng.randint(0, 1)  # hull-to-hull, almost touching

        # Image boundary check
        rh, rw = rotated.shape[:2]
        if (cx - rw // 2 < 0 or cx + rw // 2 >= image_size
                or cy - rh // 2 < 0 or cy + rh // 2 >= image_size):
            break  # cluster has fallen off the edge — stop here

        # Water check + occupancy check against OTHER events (not this cluster).
        # Intra-cluster adjacency is intentional; inter-cluster overlap is not.
        half_bw = max(1, bw // 2)
        half_lh = max(1, lh // 2)
        cy0 = max(0, cy - half_lh)
        cy1 = min(image_size, cy + half_lh)
        cx0 = max(0, cx - half_bw)
        cx1 = min(image_size, cx + half_bw)
        if not water_mask[cy0:cy1, cx0:cx1].any():
            continue
        if pre_occupancy[cy0:cy1, cx0:cx1].any():
            continue

        # Porter-Duff source-over onto cluster buffer (no background yet)
        _composite_rgba(cluster_buf, rotated, cx - rw // 2, cy - rh // 2)
        cid = _ship_class_id(lh, resolution_m, class_id, size_threshold)
        placed.append((cx, cy, bw, lh, angle_rad, cid))
        # Update occupancy immediately to prevent within-cluster overlap
        _stamp_occupancy(occupancy, cx, cy, bw, lh, angle_rad)

    # Blend the combined cluster layer onto the background once.
    if placed:
        _blend_rgba_layer(background, cluster_buf, cluster_alpha, water_tint)
        for cx, cy, bw, lh, angle_rad, cid in placed:
            corners = compute_obb_corners(
                float(cx), float(cy), float(bw), float(lh), angle_rad,
            )
            labels.append(format_obb_label(cid, corners, image_size, image_size))

    return labels


def _sample_water_tint(
    background: NDArray[np.uint8],
    cx: int,
    cy: int,
    radius: int = 8,
) -> NDArray[np.float32]:
    """Sample average water colour around a point for blending tint."""
    h, w = background.shape[:2]
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius)
    region = background[y0:y1, x0:x1]
    if region.size == 0:
        return np.array([40.0, 50.0, 60.0], dtype=np.float32)
    return region.mean(axis=(0, 1)).astype(np.float32)


# ── False-negative tile extraction ────────────────────────────────────────


def _false_source_grid(
    path: Path,
    image_size: int,
    resolution: float | None,
    geo_scale: float | None,
) -> tuple[int, int, int] | None:
    """Return ``(src_tile, n_cols, n_rows)`` for a false-negative source image.

    *src_tile* is the number of source pixels per output tile edge, computed
    with the same logic as :func:`_compose_one`.  Returns ``None`` when the
    image is too small for even one non-overlapping tile.
    """
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            with rasterio.open(path) as src:
                w, h = src.width, src.height
                if geo_scale is not None:
                    src_tile = max(1, round(image_size * geo_scale))
                elif resolution is not None:
                    native_res = (src.res[0] + src.res[1]) / 2.0
                    if src.crs and src.crs.is_geographic:
                        center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                        native_res = (
                            native_res
                            * 111320.0
                            * math.cos(math.radians(center_lat))
                        )
                    src_tile = max(1, round(image_size * resolution / native_res))
                else:
                    src_tile = image_size
        else:
            with Image.open(path) as img:
                w, h = img.size
            src_tile = image_size
    except Exception:
        logger.warning(
            "Cannot open false-negative source %s — skipping",
            path.name,
            exc_info=True,
        )
        return None

    n_cols = w // src_tile
    n_rows = h // src_tile
    if n_cols == 0 or n_rows == 0:
        logger.debug(
            "Source %s too small for %d px tiles — skipping", path.name, image_size
        )
        return None
    return src_tile, n_cols, n_rows


def generate_false_negatives(
    false_dir: Path,
    output_dir: Path,
    count: int,
    image_size: int,
    *,
    resolution: float | None = None,
    geo_scale: float | None = None,
    rng: random.Random,
    start_index: int = 0,
) -> int:
    """Write *count* false-negative (label-free) tiles from *false_dir*.

    Tiles are cropped from PNG and TIFF images found in *false_dir*.
    No two tiles from the same source image overlap (grid-based non-overlapping
    crop).  Tiles are distributed as evenly as possible across all source
    images so that no single image contributes a disproportionate share.

    Parameters
    ----------
    false_dir
        Directory containing false-negative source images (PNG / TIFF).
    output_dir
        Root of the YOLO dataset (same as :func:`generate_dataset`'s
        ``output_dir``).  Images are saved to ``images/train/`` and empty
        label files to ``labels/train/``.
    count
        Number of false-negative tiles to generate.
    image_size
        Output tile edge size in pixels.
    resolution
        Target resolution in m/px (same as :func:`generate_dataset`).
    geo_scale
        Fixed pixel scale for TIFF images (same as :func:`generate_dataset`).
    rng
        Seeded random number generator for reproducibility.
    start_index
        File index offset to avoid name clashes with the synthetic images.

    Returns
    -------
    int
        Number of tiles actually written.  May be less than *count* if the
        source images do not have enough total non-overlapping capacity.
    """
    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"

    # Collect PNG + TIFF source files (deduplicated, sorted for reproducibility)
    sources_raw: list[Path] = []
    for pattern in ("*.png", "*.tif", "*.tiff", "*.PNG", "*.TIF", "*.TIFF"):
        sources_raw.extend(false_dir.glob(pattern))
    seen: set[Path] = set()
    sources: list[Path] = []
    for p in sorted(sources_raw):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            sources.append(p)

    if not sources:
        msg = f"No PNG/TIFF images found in {false_dir}"
        raise FileNotFoundError(msg)

    # Compute grid capacity for each source
    source_grids: list[tuple[Path, int, int, int]] = []  # path, src_tile, cols, rows
    for path in sources:
        info = _false_source_grid(path, image_size, resolution, geo_scale)
        if info is not None:
            src_tile, cols, rows = info
            source_grids.append((path, src_tile, cols, rows))

    if not source_grids:
        msg = (
            f"No usable source images in {false_dir} "
            f"(all too small for {image_size}px tiles)"
        )
        raise ValueError(msg)

    n_sources = len(source_grids)
    # Cap the per-source contribution so tiles are spread across images.
    max_per_source = math.ceil(count / n_sources)

    # First pass: allocate ≤ max_per_source tiles from each source.
    allocations: list[tuple[Path, int, int, int, int]] = []  # path, src_tile, cols, rows, alloc
    remaining = count
    for path, src_tile, cols, rows in source_grids:
        alloc = min(cols * rows, max_per_source, remaining)
        allocations.append((path, src_tile, cols, rows, alloc))
        remaining -= alloc

    # Second pass: use spare capacity to cover any shortfall
    # (e.g. some sources were smaller than max_per_source).
    if remaining > 0:
        for i, (path, src_tile, cols, rows, alloc) in enumerate(allocations):
            extra = min(cols * rows - alloc, remaining)
            if extra > 0:
                allocations[i] = (path, src_tile, cols, rows, alloc + extra)
                remaining -= extra
            if remaining <= 0:
                break

    if remaining > 0:
        total_cap = sum(c * r for _, _, c, r in source_grids)
        logger.warning(
            "False-negative sources only provide %d non-overlapping tiles "
            "(requested %d). Tiles will be repeated to reach the target count.",
            total_cap,
            count,
        )
        # Third pass: distribute remaining tiles round-robin, allowing repeats.
        source_idx = 0
        while remaining > 0:
            path, src_tile, cols, rows, alloc = allocations[source_idx % n_sources]
            allocations[source_idx % n_sources] = (path, src_tile, cols, rows, alloc + 1)
            remaining -= 1
            source_idx += 1

    total_to_write = sum(a for *_, a in allocations)
    total_written = 0
    idx = start_index
    tiff_suffixes = {".tif", ".tiff"}

    with tqdm(
        total=total_to_write,
        desc="False negatives",
        unit="tile",
        dynamic_ncols=True,
    ) as pbar:
        for path, src_tile, cols, rows, alloc in allocations:
            if alloc <= 0:
                continue

            # Build grid positions and shuffle for random selection.
            grid = [(c, r) for r in range(rows) for c in range(cols)]
            rng.shuffle(grid)
            cap = len(grid)
            if alloc <= cap:
                positions = grid[:alloc]
            else:
                # Need more tiles than non-overlapping capacity: cycle with
                # re-shuffling at each full cycle for visual variety.
                positions = []
                cycle = list(grid)
                for k in range(alloc):
                    if k > 0 and k % cap == 0:
                        rng.shuffle(cycle)
                    positions.append(cycle[k % cap])

            if path.suffix.lower() in tiff_suffixes:
                with rasterio.open(path) as src:
                    for col, row in positions:
                        x0, y0 = col * src_tile, row * src_tile
                        data = src.read(
                            list(range(1, min(src.count, 3) + 1)),
                            window=Window(x0, y0, src_tile, src_tile),
                        )
                        tile_img = Image.fromarray(
                            np.moveaxis(data, 0, -1).astype(np.uint8)
                        ).convert("RGB")
                        if src_tile != image_size:
                            tile_img = tile_img.resize(
                                (image_size, image_size), Image.BILINEAR
                            )
                        name = f"{idx:06d}"
                        tile_img.save(img_out / f"{name}.png")
                        (lbl_out / f"{name}.txt").write_text("", encoding="utf-8")
                        idx += 1
                        total_written += 1
                        pbar.update(1)
            else:
                with Image.open(path) as src_img:
                    src_rgb = src_img.convert("RGB")
                    for col, row in positions:
                        x0, y0 = col * src_tile, row * src_tile
                        tile_img = src_rgb.crop(
                            (x0, y0, x0 + src_tile, y0 + src_tile)
                        )
                        if src_tile != image_size:
                            tile_img = tile_img.resize(
                                (image_size, image_size), Image.BILINEAR
                            )
                        name = f"{idx:06d}"
                        tile_img.save(img_out / f"{name}.png")
                        (lbl_out / f"{name}.txt").write_text("", encoding="utf-8")
                        idx += 1
                        total_written += 1
                        pbar.update(1)

    logger.info(
        "False negatives written: %d / %d requested", total_written, count
    )
    return total_written


# ── Main entry point ──────────────────────────────────────────────────────


def generate_dataset(
    bg_dir: Path | str,
    output_dir: Path | str,
    count: int,
    *,
    ship_dir: Path | str | None = None,
    image_size: int = 640,
    resolution: float | None = None,
    geo_scale: float | None = None,
    ships_per_image: tuple[int, int] = (0, 10),
    cluster_prob: float = 0.15,
    cluster_size: tuple[int, int] = (2, 5),
    cluster_mixed_prob: float = 0.5,
    class_id: int = 0,
    erode_coast: int = 3,
    min_water_ratio: float = 0.3,
    ship_blur_sigma: float = 0.8,
    ship_alpha: tuple[float, float] = (0.7, 0.95),
    ship_length_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
    seed: int | None = None,
    size_threshold: float | None = None,
    wake_prob_scale: float = 1.0,
    wake_alpha_scale: float = 1.0,
    max_workers: int | None = None,
    false_dir: Path | str | None = None,
    false_ratio: float = 0.0,
    coastline: Path | str | None = None,
) -> dict[str, int]:
    """Generate a synthetic ship detection dataset in YOLO OBB format.

    Parameters
    ----------
    bg_dir
        Directory containing Sentinel-2 ``*_visual.tif`` background images.
        Corresponding ``*_SCL_20m.tif`` files, if present, are used for
        water masking.
    output_dir
        Root of the output YOLO dataset.
    count
        Number of training images to generate.
    ship_dir
        Directory of pre-generated SVG ship files.  When *None*, ships are
        generated on-the-fly with :mod:`medetect.shipgen`.
    image_size
        Output tile edge size in pixels.
    resolution
        Target resolution in m/px.  *None* uses the native GeoTIFF resolution.
        When *geo_scale* is set, this is used only for ship size calculation.
    geo_scale
        When set, ignore the TIFF's geographic CRS and use a fixed pixel
        scale instead.  ``1.0`` = 1 TIFF pixel per output pixel,
        ``2.0`` = 2 TIFF pixels per output pixel (zoom out / more area),
        ``0.5`` = upsample 2× (zoom in).  ``resolution`` still controls ship
        sizes in metres.
    ships_per_image
        ``(min, max)`` number of ships per tile.
    cluster_prob
        Probability that a ship group forms a side-by-side cluster.
    cluster_size
        ``(min, max)`` ships in a cluster.
    cluster_mixed_prob
        Probability that a cluster contains mixed ship types and sizes
        rather than uniform sister ships.  ``0.0`` = always uniform,
        ``1.0`` = always mixed (default: 0.5).
    class_id
        YOLO class ID for all ships.
    erode_coast
        Pixels to erode from the water mask boundary.
    min_water_ratio
        Minimum fraction of water in a tile to be usable.
    ship_blur_sigma
        Gaussian blur sigma applied to rendered ships.
    ship_alpha
        ``(min, max)`` alpha factor range for ship blending.
    ship_length_range
        Global ``(min_m, max_m)`` constraint on ship length in metres.
        Applied on top of the per-class range.  *None* = no global limit.
    length_exponent
        Controls the size-frequency distribution.  ``1.0`` = log-uniform
        (default), ``> 1.0`` = more small ships, ``< 1.0`` = more uniform.
    seed
        Random seed for reproducibility.
    size_threshold
        Ship length threshold in metres for two-class labelling.  Ships
        shorter than this are labelled ``ship_small`` (class *class_id*),
        ships at or above are ``ship_large`` (class ``class_id + 1``).
        *None* keeps the single ``ship`` class (backward compatible).
    max_workers
        Number of parallel worker threads.  ``None`` uses ``os.cpu_count()``.
    coastline
        Path to an OSM coastline shapefile (``lines.shp``).  When set,
        coastline geometries are used to create precise water/land
        boundaries, combined with the RGB or SCL water mask via AND.

    Returns
    -------
    dict[str, int]
        Statistics: ``images``, ``ships``, ``clusters``, ``skipped``.
    """
    bg_dir = Path(bg_dir)
    output_dir = Path(output_dir)

    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Compute split: count is total images (synth + false negatives).
    if false_dir is not None and false_ratio > 0.0:
        if not (0.0 < false_ratio < 1.0):
            msg = f"false_ratio must be in (0, 1), got {false_ratio}"
            raise ValueError(msg)
        false_count = round(count * false_ratio)
        synth_count = count - false_count
    else:
        false_count = 0
        synth_count = count

    # Collect inputs
    visual_files = sorted(bg_dir.glob("*_visual.tif"))
    if not visual_files:
        visual_files = sorted(bg_dir.glob("*.tif"))
    if not visual_files:
        msg = f"No TIF files found in {bg_dir}"
        raise FileNotFoundError(msg)

    # Validate SVG dir now (before spawning workers) so errors surface early.
    svg_dir: Path | None = None
    if ship_dir is not None:
        svg_dir = Path(ship_dir)
        if not any(svg_dir.glob("*.svg")):
            msg = f"No SVG files found in {svg_dir}"
            raise FileNotFoundError(msg)

    # Validate coastline shapefile.
    coastline_path: Path | None = None
    if coastline is not None:
        coastline_path = Path(coastline)
        if not coastline_path.exists():
            msg = f"Coastline shapefile not found: {coastline_path}"
            raise FileNotFoundError(msg)

    # Pre-generate per-task parameters using the main RNG so that results
    # are reproducible with the same seed regardless of worker count.
    task_tifs = [rng.choice(visual_files) for _ in range(synth_count)]
    task_seeds = [rng.randint(0, 2**32 - 1) for _ in range(synth_count)]

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    stats = {"images": 0, "ships": 0, "clusters": 0, "skipped": 0}

    # Build shared kwargs once — svg_metas is intentionally excluded:
    # it is loaded once per worker process via _worker_init to avoid
    # pickling the full SVG list with every task submission.
    _shared = dict(
        img_out=img_out,
        lbl_out=lbl_out,
        image_size=image_size,
        resolution=resolution,
        geo_scale=geo_scale,
        ships_per_image=ships_per_image,
        cluster_prob=cluster_prob,
        cluster_size=cluster_size,
        cluster_mixed_prob=cluster_mixed_prob,
        class_id=class_id,
        erode_coast=erode_coast,
        min_water_ratio=min_water_ratio,
        ship_blur_sigma=ship_blur_sigma,
        ship_alpha=ship_alpha,
        ship_length_range=ship_length_range,
        length_exponent=length_exponent,
        size_threshold=size_threshold,
        wake_prob_scale=wake_prob_scale,
        wake_alpha_scale=wake_alpha_scale,
    )

    # Limit in-flight futures to avoid flooding the IPC queue with thousands
    # of pickled task args simultaneously.  2× workers gives enough headroom
    # to keep all workers busy without unbounded memory growth.
    max_inflight = max_workers * 2

    task_iter = iter(range(synth_count))
    pending: set[concurrent.futures.Future[tuple[int, int]]] = set()
    future_info: dict[concurrent.futures.Future[tuple[int, int]], tuple[int, Path]] = {}

    def _submit_next() -> bool:
        try:
            i = next(task_iter)
        except StopIteration:
            return False
        fut = executor.submit(
            _run_compose_task,
            index=i,
            task_seed=task_seeds[i],
            tif_path=task_tifs[i],
            **_shared,
        )
        pending.add(fut)
        future_info[fut] = (i, task_tifs[i])
        return True

    def _collect_done(pbar: tqdm) -> None:  # type: ignore[type-arg]
        done, _ = concurrent.futures.wait(
            pending, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for fut in done:
            pending.discard(fut)
            _idx, tif_path = future_info.pop(fut)
            try:
                n_ships, n_clusters = fut.result()
                if n_ships < 0:
                    stats["skipped"] += 1
                else:
                    stats["images"] += 1
                    stats["ships"] += n_ships
                    stats["clusters"] += n_clusters
            except Exception:
                logger.warning(
                    "Failed to compose %s — skipping",
                    tif_path.name,
                    exc_info=True,
                )
                stats["skipped"] += 1
            finally:
                pbar.update(1)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(svg_dir, coastline_path),
    ) as executor:
        with tqdm(
            total=synth_count,
            desc="Generating dataset",
            unit="image",
            dynamic_ncols=True,
        ) as pbar:
            # Fill the initial window.
            for _ in range(min(max_inflight, synth_count)):
                _submit_next()

            # Rolling-window: for each completed task, submit one new task.
            while pending:
                _collect_done(pbar)
                while len(pending) < max_inflight:
                    if not _submit_next():
                        break

    # Write dataset YAML
    gen_params: dict[str, object] = {
        "count": count,
        "image_size": image_size,
        "resolution": resolution,
        "geo_scale": geo_scale,
        "ships_per_image": f"{ships_per_image[0]}:{ships_per_image[1]}",
        "cluster_prob": cluster_prob,
        "cluster_size": f"{cluster_size[0]}:{cluster_size[1]}",
        "class_id": class_id,
        "erode_coast": erode_coast,
        "min_water_ratio": min_water_ratio,
        "cluster_mixed_prob": cluster_mixed_prob,
        "ship_blur_sigma": ship_blur_sigma,
        "ship_alpha": f"{ship_alpha[0]}:{ship_alpha[1]}",
        "ship_length_range": (
            f"{ship_length_range[0]}:{ship_length_range[1]}"
            if ship_length_range is not None
            else None
        ),
        "length_exponent": length_exponent,
        "seed": seed,
        "size_threshold": size_threshold,
        "wake_prob_scale": wake_prob_scale,
        "wake_alpha_scale": wake_alpha_scale,
        "false_dir": str(false_dir) if false_dir is not None else None,
        "false_ratio": false_ratio,
        "coastline": str(coastline_path) if coastline_path is not None else None,
    }
    _write_dataset_yaml(
        output_dir, class_id,
        size_threshold=size_threshold, params=gen_params,
    )

    # Generate false-negative (background-only) tiles when requested.
    stats["false_negatives"] = 0
    if false_count > 0:
        stats["false_negatives"] = generate_false_negatives(
            false_dir=Path(false_dir),  # type: ignore[arg-type]
            output_dir=output_dir,
            count=false_count,
            image_size=image_size,
            resolution=resolution,
            geo_scale=geo_scale,
            rng=rng,
            start_index=synth_count,
        )

    logger.info("Dataset complete: %s", stats)
    return stats


# ── Single image composition ──────────────────────────────────────────────


def _run_compose_task(
    *,
    index: int,
    task_seed: int,
    tif_path: Path,
    img_out: Path,
    lbl_out: Path,
    image_size: int,
    resolution: float | None,
    geo_scale: float | None,
    ships_per_image: tuple[int, int],
    cluster_prob: float,
    cluster_size: tuple[int, int],
    cluster_mixed_prob: float,
    class_id: int,
    erode_coast: int,
    min_water_ratio: float,
    ship_blur_sigma: float,
    ship_alpha: tuple[float, float],
    ship_length_range: tuple[float, float] | None,
    length_exponent: float,
    size_threshold: float | None,
    wake_prob_scale: float = 1.0,
    wake_alpha_scale: float = 1.0,
) -> tuple[int, int]:
    """Worker function for one dataset image.

    Returns ``(n_ships, n_clusters)``.  Returns ``(-1, -1)`` when the tile
    was skipped (no suitable water region found).  Raises on hard errors.

    ``svg_metas`` and ``coastline_index`` are read from process-local
    globals set by ``_worker_init`` to avoid re-pickling every call.
    """
    rng = random.Random(task_seed)
    result = _compose_one(
        tif_path=tif_path,
        svg_metas=_worker_svg_metas,
        image_size=image_size,
        resolution=resolution,
        geo_scale=geo_scale,
        ships_per_image=ships_per_image,
        cluster_prob=cluster_prob,
        cluster_size=cluster_size,
        cluster_mixed_prob=cluster_mixed_prob,
        class_id=class_id,
        erode_coast=erode_coast,
        min_water_ratio=min_water_ratio,
        ship_blur_sigma=ship_blur_sigma,
        ship_alpha=ship_alpha,
        ship_length_range=ship_length_range,
        length_exponent=length_exponent,
        rng=rng,
        size_threshold=size_threshold,
        wake_prob_scale=wake_prob_scale,
        wake_alpha_scale=wake_alpha_scale,
        coastline_index=_worker_coastline_index,
    )

    if result is None:
        return -1, -1

    tile, labels, n_clusters = result
    name = f"{index:06d}"
    Image.fromarray(tile).save(img_out / f"{name}.png")
    (lbl_out / f"{name}.txt").write_text(
        "\n".join(labels) + ("\n" if labels else ""),
        encoding="utf-8",
    )
    return len(labels), n_clusters


def _compose_one(
    *,
    tif_path: Path,
    svg_metas: list[_SvgMeta] | None,
    image_size: int,
    resolution: float | None,
    geo_scale: float | None,
    ships_per_image: tuple[int, int],
    cluster_prob: float,
    cluster_size: tuple[int, int],
    cluster_mixed_prob: float = 0.5,
    class_id: int = 0,
    erode_coast: int = 3,
    min_water_ratio: float = 0.3,
    ship_blur_sigma: float = 0.8,
    ship_alpha: tuple[float, float] = (0.7, 0.95),
    ship_length_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
    rng: random.Random,
    max_crop_attempts: int = 20,
    size_threshold: float | None = None,
    wake_prob_scale: float = 1.0,
    wake_alpha_scale: float = 1.0,
    coastline_index: CoastlineIndex | None = None,
) -> tuple[NDArray[np.uint8], list[str], int] | None:
    """Compose one training image.  Returns ``(tile, labels, n_clusters)``."""
    with rasterio.open(tif_path) as src:
        if geo_scale is not None:
            # Ignore geographic CRS; use a fixed pixel scale.
            # geo_scale=1.0 → 1 TIFF px = 1 output px
            # geo_scale=2.0 → 2 TIFF px = 1 output px (zoom out)
            src_tile = max(1, round(image_size * geo_scale))
            ship_resolution = resolution if resolution is not None else 10.0
        else:
            native_res = (src.res[0] + src.res[1]) / 2.0

            # Handle geographic CRS (degree units)
            if src.crs and src.crs.is_geographic:
                center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                native_res = native_res * 111320.0 * math.cos(math.radians(center_lat))

            if resolution is not None:
                src_tile = max(1, round(image_size * resolution / native_res))
            else:
                src_tile = image_size
                resolution = native_res
            ship_resolution = resolution  # type: ignore[assignment]

        for _ in range(max_crop_attempts):
            if src.width <= src_tile or src.height <= src_tile:
                return None
            col = rng.randint(0, src.width - src_tile)
            row = rng.randint(0, src.height - src_tile)

            try:
                tile = _read_tile(src, col, row, src_tile)
            except rasterio.errors.RasterioIOError:
                logger.debug(
                    "Tile read error in %s at col=%d row=%d — retrying",
                    tif_path.name, col, row,
                )
                continue

            # Resize if we read a different size from the output
            if src_tile != image_size:
                img = Image.fromarray(tile)
                img = img.resize((image_size, image_size), Image.BILINEAR)
                tile = np.array(img)

            # Skip satellite blackout strips (帯状の真っ黒領域)
            if is_dark_tile(tile):
                logger.debug(
                    "Dark tile in %s at col=%d row=%d (mean=%.1f) — retrying",
                    tif_path.name, col, row, float(tile.mean()),
                )
                continue

            # Water mask
            scl_file = _scl_path_for(tif_path)
            scl = _read_scl_tile(scl_file, col, row, src_tile, image_size)
            if scl is not None:
                water_mask = make_water_mask_from_scl(scl)
            else:
                water_mask = make_water_mask_from_rgb(tile)

            # Exclude no-data (pure black, #000000) pixels — artificially masked
            # or unimaged regions must not be used for ship placement.
            water_mask &= ~make_nodata_mask(tile)

            # Coastline-based mask (precise land/water boundary from OSM)
            if coastline_index is not None:
                window = Window(col, row, src_tile, src_tile)
                tile_transform = src.window_transform(window)
                if src_tile != image_size:
                    bounds = rasterio.transform.array_bounds(
                        src_tile, src_tile, tile_transform,
                    )
                    tile_transform = rasterio.transform.from_bounds(
                        *bounds, image_size, image_size,
                    )
                tile_bounds = rasterio.transform.array_bounds(
                    image_size, image_size, tile_transform,
                )
                coastline_geoms = coastline_index.query(tile_bounds)
                coastline_mask = make_water_mask_from_coastline(
                    coastline_geoms, tile, tile_transform,
                    image_size, image_size,
                )
                water_mask &= coastline_mask

            water_mask = erode_mask(water_mask, erode_coast)

            water_ratio = water_mask.sum() / water_mask.size
            if water_ratio >= min_water_ratio:
                break
        else:
            # Land-only tile — output as negative example (no ships)
            try:
                return tile, [], 0  # type: ignore[possibly-undefined]
            except NameError:
                return None

    # Colour augmentation — applied AFTER water-mask computation to avoid
    # shifting pixel values that the mask heuristic expects, but BEFORE
    # ship compositing so ships are blended onto the augmented background.
    tile = augment_tile(tile, rng)

    # Place ships
    # ships_per_image は「配置イベント数」= 単独船1隻またはクラスタ1グループ を何回行うか。
    n_events = rng.randint(*ships_per_image)
    occupancy = np.zeros((image_size, image_size), dtype=bool)
    labels: list[str] = []
    n_clusters = 0

    # Collect single-ship placements; wakes are rendered in a separate pass
    # so that no hull is obscured by a later ship's wake (z-ordering fix).
    # Each entry: (cx, cy, rotated, bw, lh, angle_rad, alpha, water_tint, state, cid, corners)
    single_ships: list[tuple] = []

    for _ in range(n_events):
        is_cluster = rng.random() < cluster_prob

        if is_cluster:
            new_labels = _place_cluster(
                water_mask, occupancy, svg_metas, ship_resolution, rng,
                cluster_size, ship_blur_sigma, ship_alpha,
                class_id, image_size, tile, ship_length_range,
                length_exponent, size_threshold,
                mixed_prob=cluster_mixed_prob,
            )
            labels.extend(new_labels)
            if new_labels:
                n_clusters += 1
        else:
            svg_text = _pick_svg(svg_metas, rng, ship_length_range)
            angle_deg = rng.uniform(0, 360)
            angle_rad = math.radians(angle_deg)
            # Render ship with rotation applied during SVG rasterization
            rotated, cls_name, bw, lh, lb = _render_ship(
                svg_text, ship_resolution, rng, ship_blur_sigma, ship_length_range,
                angle_deg=angle_deg, length_exponent=length_exponent,
            )

            available = water_mask & ~occupancy
            pos = find_water_position(available, bw, lh, angle_rad, rng)
            if pos is not None:
                cx, cy = pos
                alpha = rng.uniform(*ship_alpha)
                water_tint = _sample_water_tint(tile, cx, cy)
                ship_state = pick_motion_state(rng)
                _stamp_occupancy(occupancy, cx, cy, bw, lh, angle_rad)
                corners = compute_obb_corners(
                    float(cx), float(cy), float(bw), float(lh), angle_rad,
                )
                cid = _ship_class_id(lh, ship_resolution, class_id, size_threshold)
                single_ships.append(
                    (cx, cy, rotated, bw, lh, angle_rad, alpha, water_tint,
                     ship_state, cid, corners)
                )

    # Pass 2: render wakes for all single ships before any hull is blended.
    # This guarantees every hull appears on top of every wake (incl. its own).
    for cx, cy, rotated, bw, lh, angle_rad, alpha, water_tint, ship_state, cid, corners in single_ships:
        render_wake(
            tile, water_mask,
            float(cx), float(cy), bw, lh, angle_rad,
            ship_state, rng,
            wake_prob_scale=wake_prob_scale,
            wake_alpha_scale=wake_alpha_scale,
        )

    # Pass 3: blend all single ship hulls on top of the wakes.
    for cx, cy, rotated, bw, lh, angle_rad, alpha, water_tint, ship_state, cid, corners in single_ships:
        blend_ship(tile, rotated, cx, cy, alpha, water_tint)
        labels.append(format_obb_label(cid, corners, image_size, image_size))

    return tile, labels, n_clusters


# ── Dataset YAML ──────────────────────────────────────────────────────────


def _ship_class_id(
    length_px: int,
    resolution_m: float,
    class_id: int,
    size_threshold: float | None,
) -> int:
    """Return the YOLO class ID for a ship based on its physical length.

    When *size_threshold* is ``None``, all ships get *class_id*.  When set,
    ships shorter than the threshold get *class_id* (small) and ships at or
    above get ``class_id + 1`` (large).
    """
    if size_threshold is None:
        return class_id
    length_m = length_px * resolution_m
    if length_m >= size_threshold:
        return class_id + 1  # large
    return class_id  # small


def _write_dataset_yaml(
    output_dir: Path,
    class_id: int,
    *,
    size_threshold: float | None = None,
    params: dict[str, object] | None = None,
) -> None:
    """Write a YOLO dataset YAML config.

    When *size_threshold* is set, two classes (``ship_small`` / ``ship_large``)
    are written.  Generation parameters in *params* are prepended as YAML
    comments for reproducibility.
    """
    yaml_path = output_dir / "dataset.yaml"
    lines: list[str] = []

    # Generation parameters as comments
    if params:
        lines.append("# Generation parameters:")
        for key, value in params.items():
            lines.append(f"#   {key}: {value}")
        lines.append("")

    lines.append(f"path: {output_dir.resolve().as_posix()}")
    lines.append("train: images/autosplit_train.txt")
    lines.append("val: images/autosplit_val.txt")
    lines.append("")
    lines.append("names:")
    if size_threshold is not None:
        lines.append(f"  {class_id}: ship_small")
        lines.append(f"  {class_id + 1}: ship_large")
    else:
        lines.append(f"  {class_id}: ship")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
