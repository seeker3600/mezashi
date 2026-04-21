"""Shadow preview generation for manual QA."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from medetect.datagen.render import rasterize_ship_svg
from medetect.datagen.scene import (
    _make_shadow_rgba,
    _sample_shadow_alpha,
    _sample_water_tint,
    _shadow_alpha_for_ship,
    _shadow_blur_sigma,
    _shadow_offset_pixels,
    blend_shadow,
    blend_ship,
)
from medetect.shipgen.gen import generate_ship_svg


def _make_ocean_background(height: int, width: int, rng: random.Random) -> np.ndarray:
    base = np.array([45, 78, 96], dtype=np.float32)
    background = np.full((height, width, 3), base, dtype=np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    for channel in range(3):
        raw = np_rng.random((height // 10, width // 10)).astype(np.float32)
        noise_img = Image.fromarray((raw * 255).astype(np.uint8), "L")
        noise_img = noise_img.resize((width, height), Image.BILINEAR)
        noise = np.asarray(noise_img).astype(np.float32) / 255.0
        background[:, :, channel] += (noise - 0.5) * (6 + channel * 2)
    return np.clip(background, 0, 255).astype(np.uint8)


def _make_flat_background(height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:, :] = color
    return background


def _draw_ship_outlines(background: np.ndarray, placements: list[tuple[int, int, np.ndarray, str]]) -> None:
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    for cx, cy, ship_rgba, label in placements:
        height, width = ship_rgba.shape[:2]
        x0 = cx - width // 2
        y0 = cy - height // 2
        draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(255, 255, 255), width=1)
        draw.text((x0 + 2, y0 - 14), label, fill=(255, 248, 180))
    background[:] = np.asarray(image)


def _render_panel(
    base_background: np.ndarray,
    specs: list[tuple[str, int, int, int, int, int]],
    *,
    shadow_azimuth_rad: float,
    shadow_length: float | None,
    shadow_alpha: float = 0.1,
    shadow_alpha_scale: float = 1.0,
) -> np.ndarray:
    background = base_background.copy()
    water_mask = np.ones(background.shape[:2], dtype=bool)
    placements: list[tuple[int, int, np.ndarray, str]] = []

    for index, (ship_class, cx, cy, beam_px, length_px, angle_deg) in enumerate(specs):
        svg = generate_ship_svg(ship_class, rng=random.Random(1000 + index))
        ship_rgba = rasterize_ship_svg(svg, beam_px, length_px, angle_deg=angle_deg, supersample=4)
        water_tint = _sample_water_tint(background, cx, cy)
        if shadow_length is not None:
            offset_x, offset_y = _shadow_offset_pixels(beam_px, length_px, shadow_azimuth_rad, shadow_length)
            cast_length = math.hypot(offset_x, offset_y)
            shadow_rgba = _make_shadow_rgba(
                ship_rgba,
                offset_x=offset_x,
                offset_y=offset_y,
                blur_sigma=_shadow_blur_sigma(beam_px, length_px, cast_length),
                alpha_scale=_shadow_alpha_for_ship(beam_px, length_px),
            )
            blend_shadow(
                background,
                shadow_rgba,
                cx,
                cy,
                alpha_factor=shadow_alpha * shadow_alpha_scale,
                clip_mask=water_mask,
            )
        blend_ship(background, ship_rgba, cx, cy, alpha_factor=0.9, water_tint=water_tint)
        placements.append((cx, cy, ship_rgba, f"{ship_class} {angle_deg}deg"))

    _draw_ship_outlines(background, placements)
    return background


def render_shadow_previews(output_dir: Path) -> dict[str, Path]:
    """Generate a standard set of shadow preview images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    size = 512
    base_background = _make_ocean_background(size, size, rng)
    white_background = _make_flat_background(size, size, (255, 255, 255))
    shadow_azimuth_rad = math.radians(25.0)
    shadow_alpha = _sample_shadow_alpha(random.Random(7))
    shadow_lengths = [0.0, 0.8, 1.8, 3.5]
    alpha_scales = [0.5, 1.0, 1.5, 2.0]

    specs = [
        ("destroyer", 90, 85, 12, 72, 5),
        ("frigate", 220, 105, 18, 90, 40),
        ("carrier", 390, 95, 24, 136, 85),
        ("barge", 105, 275, 18, 105, 125),
        ("fishing_trawler", 255, 260, 12, 62, 180),
        ("tug_harbor", 390, 290, 10, 50, 235),
        ("destroyer", 145, 420, 14, 82, 300),
        ("frigate", 350, 405, 16, 88, 340),
    ]

    plain = _render_panel(base_background, specs, shadow_azimuth_rad=shadow_azimuth_rad, shadow_length=None, shadow_alpha=shadow_alpha)
    panels = [
        _render_panel(
            base_background,
            specs,
            shadow_azimuth_rad=shadow_azimuth_rad,
            shadow_length=shadow_length,
            shadow_alpha=shadow_alpha,
        )
        for shadow_length in shadow_lengths
    ]
    alpha_panels = [
        _render_panel(
            base_background,
            specs,
            shadow_azimuth_rad=shadow_azimuth_rad,
            shadow_length=1.8,
            shadow_alpha=shadow_alpha,
            shadow_alpha_scale=alpha_scale,
        )
        for alpha_scale in alpha_scales
    ]
    alpha_panels_white = [
        _render_panel(
            white_background,
            specs,
            shadow_azimuth_rad=shadow_azimuth_rad,
            shadow_length=1.8,
            shadow_alpha=shadow_alpha,
            shadow_alpha_scale=alpha_scale,
        )
        for alpha_scale in alpha_scales
    ]

    gap = 12
    positions = [(0, 0), (size + gap, 0), (0, size + gap), (size + gap, size + gap)]

    canvas = np.full((size * 2 + gap, size * 2 + gap, 3), 22, dtype=np.uint8)
    for panel, shadow_length, (x0, y0) in zip(panels, shadow_lengths, positions, strict=True):
        canvas[y0 : y0 + size, x0 : x0 + size] = panel
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.text((x0 + 10, y0 + 10), f"len={shadow_length:.2f}", fill=(255, 245, 190))
        canvas[:] = np.asarray(img)

    alpha_canvas = np.full((size * 2 + gap, size * 2 + gap, 3), 22, dtype=np.uint8)
    for panel, alpha_scale, (x0, y0) in zip(alpha_panels, alpha_scales, positions, strict=True):
        alpha_canvas[y0 : y0 + size, x0 : x0 + size] = panel
        img = Image.fromarray(alpha_canvas)
        draw = ImageDraw.Draw(img)
        draw.text((x0 + 10, y0 + 10), f"alpha={alpha_scale:.1f}", fill=(255, 245, 190))
        alpha_canvas[:] = np.asarray(img)

    alpha_white_canvas = np.full((size * 2 + gap, size * 2 + gap, 3), 245, dtype=np.uint8)
    for panel, alpha_scale, (x0, y0) in zip(alpha_panels_white, alpha_scales, positions, strict=True):
        alpha_white_canvas[y0 : y0 + size, x0 : x0 + size] = panel
        img = Image.fromarray(alpha_white_canvas)
        draw = ImageDraw.Draw(img)
        draw.text((x0 + 10, y0 + 10), f"alpha={alpha_scale:.1f}", fill=(32, 32, 32))
        alpha_white_canvas[:] = np.asarray(img)

    diff = np.abs(panels[-1].astype(np.int16) - plain.astype(np.int16))
    diff = np.clip(diff * 4, 0, 255).astype(np.uint8)

    outputs = {
        "shadow_preview": output_dir / "shadow_preview.png",
        "shadow_preview_noshadow": output_dir / "shadow_preview_noshadow.png",
        "shadow_preview_alpha": output_dir / "shadow_preview_alpha.png",
        "shadow_preview_alpha_white": output_dir / "shadow_preview_alpha_white.png",
        "shadow_preview_diff": output_dir / "shadow_preview_diff.png",
    }
    Image.fromarray(canvas).save(outputs["shadow_preview"])
    Image.fromarray(plain).save(outputs["shadow_preview_noshadow"])
    Image.fromarray(alpha_canvas).save(outputs["shadow_preview_alpha"])
    Image.fromarray(alpha_white_canvas).save(outputs["shadow_preview_alpha_white"])
    Image.fromarray(diff).save(outputs["shadow_preview_diff"])
    return outputs
