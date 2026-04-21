"""Wake preview generation for manual QA."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from medetect.datagen.wake import MotionState, render_wake


def _make_ocean_background(height: int, width: int, rng: random.Random) -> np.ndarray:
    base = np.array([55, 85, 105], dtype=np.float32)
    background = np.full((height, width, 3), base, dtype=np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    for channel in range(3):
        raw = np_rng.random((height // 8, width // 8)).astype(np.float32)
        noise_img = Image.fromarray((raw * 255).astype(np.uint8), "L")
        noise_img = noise_img.resize((width, height), Image.BILINEAR)
        noise = np.asarray(noise_img).astype(np.float32) / 255.0
        background[:, :, channel] += (noise - 0.5) * 8
    return np.clip(background, 0, 255).astype(np.uint8)


def _draw_ship_marker(background: np.ndarray, cx: float, cy: float, beam: int, length: int, angle: float) -> None:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    half_w = beam / 2
    half_h = length / 2
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    points = [(cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a) for dx, dy in corners]
    img = Image.fromarray(background)
    draw = ImageDraw.Draw(img)
    draw.polygon([(point[0], point[1]) for point in points], outline=(255, 255, 255))
    background[:] = np.asarray(img)


def render_wake_previews(output_dir: Path) -> dict[str, Path]:
    """Generate wake preview images for manual QA."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    size = 640
    background = _make_ocean_background(size, size, rng)
    water_mask = np.ones((size, size), dtype=bool)
    background_no_wake = background.copy()

    ships = [
        (100, 80, 8, 40, 20, MotionState.FAST, "FAST"),
        (350, 60, 14, 70, 0, MotionState.FAST, "FAST large"),
        (530, 100, 5, 22, 60, MotionState.MEDIUM, "MED small"),
        (100, 300, 10, 50, 350, MotionState.MEDIUM, "MED"),
        (330, 260, 6, 30, 180, MotionState.SLOW, "SLOW"),
        (530, 300, 16, 80, 90, MotionState.FAST, "FAST big"),
        (80, 500, 5, 20, 45, MotionState.STOPPED, "STOP"),
        (250, 520, 7, 32, 210, MotionState.SLOW, "SLOW"),
        (430, 500, 10, 45, 315, MotionState.MEDIUM, "MED"),
        (580, 520, 4, 14, 120, MotionState.FAST, "FAST tiny"),
    ]

    for cx, cy, beam, length, angle_deg, state, _label in ships:
        render_wake(
            background,
            water_mask,
            float(cx),
            float(cy),
            beam,
            length,
            math.radians(angle_deg),
            state,
            random.Random(rng.randint(0, 2**31)),
            wake_prob_scale=5.0,
            wake_alpha_scale=1.0,
        )

    for cx, cy, beam, length, angle_deg, _state, _label in ships:
        _draw_ship_marker(background, cx, cy, beam, length, math.radians(angle_deg))

    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)
    for cx, cy, _beam, _length, _angle_deg, _state, label in ships:
        draw.text((cx + 10, cy - 15), label, fill=(255, 255, 100))

    diff = np.abs(background_no_wake.astype(np.int16) - np.asarray(image).astype(np.int16))
    enhanced = np.clip(diff * 5, 0, 255).astype(np.uint8)
    crop_x, crop_y, crop_size = 250, 260, 200
    zoomed = background[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]
    zoomed_img = Image.fromarray(zoomed).resize((crop_size * 4, crop_size * 4), Image.NEAREST)

    outputs = {
        "wake_preview": output_dir / "wake_preview.png",
        "wake_preview_diff": output_dir / "wake_preview_diff.png",
        "wake_preview_zoom": output_dir / "wake_preview_zoom.png",
    }
    image.save(outputs["wake_preview"])
    Image.fromarray(enhanced).save(outputs["wake_preview_diff"])
    zoomed_img.save(outputs["wake_preview_zoom"])
    return outputs
