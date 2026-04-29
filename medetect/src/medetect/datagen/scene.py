"""Scene rendering helpers for synthetic datagen."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from medetect.datagen.render import (
    downsample_rgba_premultiplied_exact,
    gaussian_blur_rgba_premultiplied,
    rasterize_ship_svg,
    resize_rgba_premultiplied,
)
from medetect.datagen.ship import _resolve_ship_dimensions
from medetect.datagen.wake import MotionState


@dataclass(frozen=True)
class SingleShipPlacement:
    """Resolved single-ship placement rendered in the 3-pass compose flow."""

    cx: int
    cy: int
    rotated: NDArray[np.uint8]
    bw: int
    lh: int
    angle_rad: float
    alpha: float
    water_tint: NDArray[np.float32]
    shadow_rgba: NDArray[np.uint8] | None
    ship_state: MotionState
    class_id: int
    corners: list[tuple[float, float]]


@dataclass(frozen=True)
class RgbaLayerPatch:
    """Top-left anchored RGBA patch in image coordinates."""

    x0: int
    y0: int
    layer: NDArray[np.uint8]


_SHADOW_TILE_ALPHA_RANGE = (0.08, 0.11)
_SHADOW_SIZE_ALPHA_BOOST_MAX = 0.12
DEFAULT_WATER_TINT_STRENGTH = 0.18


def _sample_shadow_alpha(rng: random.Random) -> float:
    """Return an image-wide base darkness factor for ship shadows."""
    return rng.uniform(*_SHADOW_TILE_ALPHA_RANGE)


def _blend_ship_rgb_with_water_tint(
    ship_rgb: NDArray[np.float32],
    water_tint: NDArray[np.float32] | None,
    water_tint_strength: float,
) -> NDArray[np.float32]:
    """Mix ship RGB toward the sampled water tint."""
    if water_tint is None or water_tint_strength <= 0.0:
        return ship_rgb

    tint_strength = float(np.clip(water_tint_strength, 0.0, 1.0))
    return ship_rgb * (1.0 - tint_strength) + water_tint * tint_strength


def blend_ship(
    background: NDArray[np.uint8],
    ship_rgba: NDArray[np.uint8],
    cx: int,
    cy: int,
    alpha_factor: float = 0.85,
    water_tint: NDArray[np.float32] | None = None,
    water_tint_strength: float = DEFAULT_WATER_TINT_STRENGTH,
) -> None:
    """Alpha-composite *ship_rgba* onto *background* centred at ``(cx, cy)``."""
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
    ship_rgb = _blend_ship_rgb_with_water_tint(
        ship_crop[:, :, :3].astype(np.float32),
        water_tint,
        water_tint_strength,
    )

    blended = bg_crop.astype(np.float32) * (1.0 - alpha) + ship_rgb * alpha
    background[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = (
        blended.clip(0, 255).astype(np.uint8)
    )


def _composite_rgba(
    dst: NDArray[np.uint8],
    src: NDArray[np.uint8],
    x0: int,
    y0: int,
) -> None:
    """Porter-Duff source-over composite *src* onto *dst* at ``(x0, y0)``."""
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]

    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw = min(sw - sx0, dw - dx0)
    ch = min(sh - sy0, dh - dy0)
    if cw <= 0 or ch <= 0:
        return

    src_crop = src[sy0 : sy0 + ch, sx0 : sx0 + cw].astype(np.float32)
    dst_crop = dst[dy0 : dy0 + ch, dx0 : dx0 + cw].astype(np.float32)

    src_alpha = src_crop[:, :, 3:4] / 255.0
    dst_alpha = dst_crop[:, :, 3:4] / 255.0

    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    safe_alpha = np.where(out_alpha > 0, out_alpha, 1.0)
    out_rgb = (
        src_crop[:, :, :3] * src_alpha
        + dst_crop[:, :, :3] * dst_alpha * (1.0 - src_alpha)
    ) / safe_alpha

    dst[dy0 : dy0 + ch, dx0 : dx0 + cw, :3] = out_rgb.clip(0, 255).astype(np.uint8)
    dst[dy0 : dy0 + ch, dx0 : dx0 + cw, 3:4] = (
        (out_alpha * 255.0).clip(0, 255).astype(np.uint8)
    )


def _blend_rgba_layer(
    background: NDArray[np.uint8],
    layer: NDArray[np.uint8],
    alpha_factor: float,
    water_tint: NDArray[np.float32] | None,
    water_tint_strength: float = DEFAULT_WATER_TINT_STRENGTH,
) -> None:
    """Alpha-composite an RGBA *layer* onto an RGB *background* in place."""
    alpha = (layer[:, :, 3:4].astype(np.float32) / 255.0) * alpha_factor
    ship_rgb = _blend_ship_rgb_with_water_tint(
        layer[:, :, :3].astype(np.float32),
        water_tint,
        water_tint_strength,
    )
    blended = background.astype(np.float32) * (1.0 - alpha) + ship_rgb * alpha
    background[:] = blended.clip(0, 255).astype(np.uint8)


def _blend_rgba_patch(
    background: NDArray[np.uint8],
    patch: RgbaLayerPatch,
    alpha_factor: float,
    water_tint: NDArray[np.float32] | None,
    water_tint_strength: float = DEFAULT_WATER_TINT_STRENGTH,
) -> None:
    """Alpha-composite an RGBA patch onto an RGB background in place."""
    ph, pw = patch.layer.shape[:2]
    bh, bw = background.shape[:2]

    sx0, sy0 = max(0, -patch.x0), max(0, -patch.y0)
    dx0, dy0 = max(0, patch.x0), max(0, patch.y0)
    cw = min(pw - sx0, bw - dx0)
    ch = min(ph - sy0, bh - dy0)
    if cw <= 0 or ch <= 0:
        return

    bg_crop = background[dy0 : dy0 + ch, dx0 : dx0 + cw]
    layer_crop = patch.layer[sy0 : sy0 + ch, sx0 : sx0 + cw]
    _blend_rgba_layer(
        bg_crop,
        layer_crop,
        alpha_factor,
        water_tint,
        water_tint_strength,
    )


def _darken_rgba_layer(
    background: NDArray[np.uint8],
    layer: NDArray[np.uint8],
    alpha_factor: float,
    clip_mask: NDArray[np.bool_] | None = None,
) -> None:
    """Darken *background* in-place using the alpha channel of *layer*."""
    alpha = (layer[:, :, 3:4].astype(np.float32) / 255.0) * alpha_factor
    if clip_mask is not None:
        alpha *= clip_mask[:, :, None].astype(np.float32)
    alpha = np.clip(alpha, 0.0, 0.98)
    darkened = background.astype(np.float32) * (1.0 - alpha)
    background[:] = darkened.clip(0, 255).astype(np.uint8)


def _darken_rgba_patch(
    background: NDArray[np.uint8],
    patch: RgbaLayerPatch,
    alpha_factor: float,
    clip_mask: NDArray[np.bool_] | None = None,
) -> None:
    """Darken a background image in-place using an RGBA patch alpha."""
    ph, pw = patch.layer.shape[:2]
    bh, bw = background.shape[:2]

    sx0, sy0 = max(0, -patch.x0), max(0, -patch.y0)
    dx0, dy0 = max(0, patch.x0), max(0, patch.y0)
    cw = min(pw - sx0, bw - dx0)
    ch = min(ph - sy0, bh - dy0)
    if cw <= 0 or ch <= 0:
        return

    bg_crop = background[dy0 : dy0 + ch, dx0 : dx0 + cw]
    layer_crop = patch.layer[sy0 : sy0 + ch, sx0 : sx0 + cw]
    mask_crop = None
    if clip_mask is not None:
        mask_crop = clip_mask[dy0 : dy0 + ch, dx0 : dx0 + cw]
    _darken_rgba_layer(bg_crop, layer_crop, alpha_factor, clip_mask=mask_crop)


def blend_shadow(
    background: NDArray[np.uint8],
    shadow_rgba: NDArray[np.uint8],
    cx: int,
    cy: int,
    alpha_factor: float = 1.0,
    clip_mask: NDArray[np.bool_] | None = None,
) -> None:
    """Darken *background* using a centred shadow RGBA patch."""
    sh, sw = shadow_rgba.shape[:2]
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

    shadow_crop = shadow_rgba[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    bg_crop = background[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w]
    mask_crop = None
    if clip_mask is not None:
        mask_crop = clip_mask[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w]
    _darken_rgba_layer(bg_crop, shadow_crop, alpha_factor, clip_mask=mask_crop)


def _shadow_offset_pixels(
    beam_px: int,
    length_px: int,
    azimuth_rad: float,
    shadow_length: float,
    *,
    scene_scale: int = 1,
) -> tuple[int, int]:
    """Return a ship-size-aware cast-shadow offset in image-space pixels."""
    effective_height_px = max(
        0.75 * scene_scale,
        (beam_px * 0.55 + length_px * 0.035) * scene_scale,
    )
    cast_length_px = effective_height_px * max(0.0, shadow_length)
    max_cast_length = max(1.0, (length_px * 2.5 + beam_px * 2.5) * scene_scale)
    cast_length_px = min(max_cast_length, cast_length_px)
    if cast_length_px < 0.5:
        return 0, 0
    return (
        round(math.cos(azimuth_rad) * cast_length_px),
        round(math.sin(azimuth_rad) * cast_length_px),
    )


def _shadow_blur_sigma(
    beam_px: int,
    length_px: int,
    cast_length_px: float,
    *,
    scene_scale: int = 1,
) -> float:
    """Return a soft-edge blur radius for a ship shadow."""
    base_blur = max(0.45, beam_px * 0.05 + length_px * 0.012) * scene_scale
    return base_blur + cast_length_px * 0.04


def _shadow_alpha_for_ship(
    beam_px: int,
    length_px: int,
) -> float:
    """Return a subtle size-based shadow boost for larger hulls."""
    size_boost = beam_px * 0.0020 + length_px * 0.00045
    return 1.0 + min(_SHADOW_SIZE_ALPHA_BOOST_MAX, size_boost)


def _stamp_shadow_alpha(
    canvas_alpha: NDArray[np.uint8],
    source_alpha: NDArray[np.float32],
    x0: int,
    y0: int,
    strength: float,
) -> None:
    """Stamp alpha into a shadow canvas using max-composition."""
    if strength <= 0.0:
        return

    sh, sw = source_alpha.shape
    dh, dw = canvas_alpha.shape

    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw = min(sw - sx0, dw - dx0)
    ch = min(sh - sy0, dh - dy0)
    if cw <= 0 or ch <= 0:
        return

    src_crop = source_alpha[sy0 : sy0 + ch, sx0 : sx0 + cw]
    stamp = np.clip(src_crop * strength, 0.0, 255.0).astype(np.uint8)
    dst_crop = canvas_alpha[dy0 : dy0 + ch, dx0 : dx0 + cw]
    np.maximum(dst_crop, stamp, out=dst_crop)


def _make_shadow_rgba(
    ship_rgba: NDArray[np.uint8],
    *,
    offset_x: int,
    offset_y: int,
    blur_sigma: float,
    alpha_scale: float,
) -> NDArray[np.uint8]:
    """Create a physically-based shadow using shift-subtract.

    The shadow is the set-difference between the hull silhouette shifted by
    the cast offset and the original hull footprint.  This naturally produces
    a crescent-shaped shadow on the side opposite the sun — no directional
    mask is needed.

    Pipeline: cast sweep → hull subtraction → Gaussian blur.
    """
    height, width = ship_rgba.shape[:2]
    if alpha_scale <= 0.0 or height == 0 or width == 0:
        return np.zeros((height, width, 4), dtype=np.uint8)

    source_alpha = ship_rgba[:, :, 3].astype(np.float32)
    if float(source_alpha.max()) <= 0.0:
        return np.zeros((height, width, 4), dtype=np.uint8)

    offset_length = math.hypot(offset_x, offset_y)

    # No visible shadow when offset is negligible (sun nearly overhead).
    if offset_length < 0.5:
        return np.zeros((height, width, 4), dtype=np.uint8)

    pad_x = abs(offset_x) + math.ceil(blur_sigma * 2.5) + 3
    pad_y = abs(offset_y) + math.ceil(blur_sigma * 2.5) + 3
    shadow = np.zeros((height + pad_y * 2, width + pad_x * 2, 4), dtype=np.uint8)
    shadow_alpha = shadow[:, :, 3]

    # --- Cast sweep: stamp shifted silhouettes (step 1..N) ---
    steps = max(2, min(24, math.ceil(offset_length)))
    for step in range(1, steps + 1):
        progress = step / steps
        step_x0 = pad_x + round(offset_x * progress)
        step_y0 = pad_y + round(offset_y * progress)
        step_strength = alpha_scale * (0.92 - 0.58 * progress)
        _stamp_shadow_alpha(
            shadow_alpha,
            source_alpha,
            step_x0,
            step_y0,
            step_strength,
        )

    # --- Hull subtraction: erase shadow under the ship footprint ---
    hull_mask = source_alpha > 0.0
    hull_region = shadow_alpha[pad_y : pad_y + height, pad_x : pad_x + width]
    hull_region[hull_mask] = 0

    # --- Gaussian blur ---
    if blur_sigma > 0.0:
        img = Image.fromarray(shadow)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        shadow = np.array(img)

    return shadow


def _render_ship(
    svg_text: str,
    resolution_m: float,
    rng: random.Random,
    blur_sigma: float,
    length_range: tuple[float, float] | None = None,
    angle_deg: float = 0.0,
    length_exponent: float = 1.0,
    supersample: int = 4,
) -> tuple[NDArray[np.uint8], str, int, int, float]:
    """Render one ship and return ``(rgba, class_name, beam_px, length_px, lb_ratio)``."""
    ship_class, beam_px, length_px, lb_ratio = _resolve_ship_dimensions(
        svg_text,
        resolution_m,
        rng,
        length_range,
        length_exponent,
    )

    rgba = rasterize_ship_svg(
        svg_text,
        beam_px,
        length_px,
        angle_deg=angle_deg,
        supersample=supersample,
    )

    if blur_sigma > 0 and min(beam_px, length_px) > 2:
        rgba = gaussian_blur_rgba_premultiplied(rgba, blur_sigma)

    return rgba, ship_class, beam_px, length_px, lb_ratio


def _rasterize_ship_scene(
    svg_text: str,
    beam_px: int,
    length_px: int,
    *,
    angle_deg: float,
    blur_sigma: float,
    scene_scale: int,
) -> NDArray[np.uint8]:
    """Rasterize one ship at cluster-scene resolution for subpixel placement."""
    rgba = rasterize_ship_svg(
        svg_text,
        max(1, beam_px * scene_scale),
        max(1, length_px * scene_scale),
        angle_deg=angle_deg,
        supersample=1,
    )
    if blur_sigma > 0 and min(beam_px, length_px) > 2:
        rgba = gaussian_blur_rgba_premultiplied(rgba, blur_sigma * scene_scale)
    return rgba


def _cluster_scene_origin(
    cx: float,
    cy: float,
    ship_rgba: NDArray[np.uint8],
    scene_scale: int,
) -> tuple[int, int]:
    """Return the top-left scene pixel for a ship centred at ``(cx, cy)``."""
    cx_scene = round(cx * scene_scale)
    cy_scene = round(cy * scene_scale)
    return cx_scene - ship_rgba.shape[1] // 2, cy_scene - ship_rgba.shape[0] // 2


def _scene_patch_bounds(
    bounds: list[tuple[float, float, float, float]],
    scene_size: int,
    scene_scale: int,
    *,
    padding: int = 0,
) -> tuple[int, int, int, int] | None:
    """Return supersample-grid-aligned scene bounds for a local cluster patch."""
    if not bounds:
        return None

    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)

    if max_x <= 0.0 or max_y <= 0.0 or min_x >= scene_size or min_y >= scene_size:
        return None

    x0 = max(0, int(math.floor((min_x - padding) / scene_scale)) * scene_scale)
    y0 = max(0, int(math.floor((min_y - padding) / scene_scale)) * scene_scale)
    x1 = min(scene_size, int(math.ceil((max_x + padding) / scene_scale)) * scene_scale)
    y1 = min(scene_size, int(math.ceil((max_y + padding) / scene_scale)) * scene_scale)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _downsample_cluster_patch(
    layer: NDArray[np.uint8],
    scene_x0: int,
    scene_y0: int,
    scene_scale: int,
) -> RgbaLayerPatch:
    """Downsample a local supersampled cluster patch to image coordinates."""
    if scene_x0 % scene_scale != 0 or scene_y0 % scene_scale != 0:
        msg = "scene patch origin must align with the supersample grid"
        raise ValueError(msg)

    if scene_scale == 1:
        return RgbaLayerPatch(scene_x0, scene_y0, layer)

    if layer.shape[0] % scene_scale != 0 or layer.shape[1] % scene_scale != 0:
        msg = "scene patch dimensions must align with the supersample grid"
        raise ValueError(msg)

    out_w = max(1, layer.shape[1] // scene_scale)
    out_h = max(1, layer.shape[0] // scene_scale)
    downsampled = downsample_rgba_premultiplied_exact(layer, scene_scale)
    return RgbaLayerPatch(scene_x0 // scene_scale, scene_y0 // scene_scale, downsampled)


def _downsample_cluster_layer(
    layer: NDArray[np.uint8],
    image_size: int,
    scene_scale: int,
) -> NDArray[np.uint8]:
    """Convert a supersampled cluster layer back to image resolution."""
    if scene_scale == 1:
        return layer
    return downsample_rgba_premultiplied_exact(layer, scene_scale)


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