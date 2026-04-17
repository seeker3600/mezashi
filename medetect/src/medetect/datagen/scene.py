"""Scene rendering helpers for synthetic datagen."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from medetect.datagen.render import rasterize_ship_svg, resize_rgba_premultiplied
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
    ship_state: MotionState
    class_id: int
    corners: list[tuple[float, float]]


def blend_ship(
    background: NDArray[np.uint8],
    ship_rgba: NDArray[np.uint8],
    cx: int,
    cy: int,
    alpha_factor: float = 0.85,
    water_tint: NDArray[np.float32] | None = None,
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
    ship_rgb = ship_crop[:, :, :3].astype(np.float32)
    if water_tint is not None:
        ship_rgb = ship_rgb * 0.82 + water_tint * 0.18

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
) -> None:
    """Alpha-composite an RGBA *layer* onto an RGB *background* in place."""
    alpha = (layer[:, :, 3:4].astype(np.float32) / 255.0) * alpha_factor
    ship_rgb = layer[:, :, :3].astype(np.float32)
    if water_tint is not None:
        ship_rgb = ship_rgb * 0.82 + water_tint * 0.18
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
        img = Image.fromarray(rgba)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        rgba = np.array(img)

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
        img = Image.fromarray(rgba)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma * scene_scale))
        rgba = np.array(img)
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


def _downsample_cluster_layer(
    layer: NDArray[np.uint8],
    image_size: int,
    scene_scale: int,
) -> NDArray[np.uint8]:
    """Convert a supersampled cluster layer back to image resolution."""
    if scene_scale == 1:
        return layer
    return resize_rgba_premultiplied(layer, image_size, image_size)


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