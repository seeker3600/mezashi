"""Beam-profile helpers for shipgen visual QA."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from medetect.datagen.render import rasterize_ship_svg
from medetect.shipgen.gen import generate_ship_svg


@dataclass(frozen=True)
class BeamProfileMetrics:
    """Summary of a beam-direction brightness profile through a rendered ship."""

    positions: NDArray[np.float32]
    brightness: NDArray[np.float32]
    left_edge_delta: float
    right_edge_delta: float
    has_dark_outline: bool
    has_bright_outline: bool
    x0_px: int
    x1_px: int
    y_px: int
    row_count: int


@dataclass(frozen=True)
class OutlineCheck:
    """Bilateral edge-outline classification for a single beam profile."""

    left_edge_delta: float
    right_edge_delta: float
    has_dark_outline: bool
    has_bright_outline: bool


def composite_rgba_on_background(
    rgba: NDArray[np.uint8],
    bg_color: tuple[int, int, int] = (40, 60, 90),
) -> NDArray[np.uint8]:
    """Composite an RGBA ship render onto a solid RGB background."""
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    background = np.empty(rgba.shape[:2] + (3,), dtype=np.uint8)
    background[:, :] = bg_color
    blended = rgb * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def sample_midship_beam_profile(
    rgba: NDArray[np.uint8],
    *,
    bg_color: tuple[int, int, int] = (40, 60, 90),
    band_center_frac: float = 0.5,
    band_height_frac: float = 0.45,
    alpha_threshold: int = 32,
    sample_count: int = 128,
) -> tuple[NDArray[np.float32], NDArray[np.float32], int, int, int, int]:
    """Sample and aggregate a beam-direction profile around the ship mid-body."""
    height, _width = rgba.shape[:2]
    center_y = int(round((height - 1) * band_center_frac))
    half_band = max(1, int(round(height * band_height_frac * 0.5)))

    rgb = composite_rgba_on_background(rgba, bg_color=bg_color)
    brightness = (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    )
    alpha = rgba[:, :, 3]

    def _outline_signature(profile: NDArray[np.float32]) -> tuple[bool, bool, float, float]:
        edge_n = max(2, int(round(len(profile) * 0.12)))
        inner_n = max(2, int(round(len(profile) * 0.18)))
        left_delta = float(profile[:edge_n].mean() - profile[edge_n : edge_n + inner_n].mean())
        right_delta = float(profile[-edge_n:].mean() - profile[-edge_n - inner_n : -edge_n].mean())
        dark = left_delta < -10.0 and right_delta < -10.0
        bright = left_delta > 10.0 and right_delta > 10.0
        return dark, bright, left_delta, right_delta

    candidates: list[tuple[float, int, int, int, NDArray[np.float32], float, float]] = []
    min_width = max(8, sample_count // 4)

    for y in range(max(0, center_y - half_band), min(height, center_y + half_band + 1)):
        xs = np.where(alpha[y] > alpha_threshold)[0]
        if xs.size < min_width:
            continue
        x0 = int(xs[0])
        x1 = int(xs[-1])
        row = brightness[y, x0 : x1 + 1]
        _dark, _bright, left_delta, right_delta = _outline_signature(row.astype(np.float32))
        src_x = np.linspace(0.0, 1.0, len(row), dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        profile = np.interp(dst_x, src_x, row).astype(np.float32)
        curvature = float(np.mean(np.abs(np.diff(profile, n=2)))) if len(profile) >= 3 else 0.0
        candidates.append((curvature, x0, x1, y, profile, left_delta, right_delta))

    if not candidates:
        raise ValueError("No opaque midship rows were found in the rendered ship image")

    max_width = max(x1 - x0 + 1 for _curvature, x0, x1, _y, _profile, _left, _right in candidates)
    wide_candidates = [
        item
        for item in candidates
        if (item[2] - item[1] + 1) >= max_width * 0.88
    ]
    selected_pool = wide_candidates or candidates
    selected_pool.sort(key=lambda item: item[0])
    keep_n = min(len(selected_pool), max(3, len(selected_pool) // 3))
    selected = selected_pool[:keep_n]
    stacked = np.stack(
        [profile for _curvature, _x0, _x1, _y, profile, _left, _right in selected],
        axis=0,
    )
    positions = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    profile = np.median(stacked, axis=0).astype(np.float32)

    best_row = min(
        selected_pool,
        key=lambda item: (
            1 if (item[5] < -10.0 and item[6] < -10.0) or (item[5] > 10.0 and item[6] > 10.0) else 0,
            item[0],
            abs(item[5]) + abs(item[6]),
        ),
    )
    _best_curvature, x0_px, x1_px, y_px, _best_profile, _left_delta, _right_delta = best_row
    return positions, profile, x0_px, y_px, x1_px, len(selected)


def profile_brightness(values: NDArray[np.float32] | NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convert grayscale or RGB profile values into 1-D brightness."""
    if values.ndim == 1:
        return values.astype(np.float32)
    if values.shape[1] < 3:
        return values.astype(np.float32).mean(axis=1)
    return (
        0.299 * values[:, 0].astype(np.float32)
        + 0.587 * values[:, 1].astype(np.float32)
        + 0.114 * values[:, 2].astype(np.float32)
    )


def summarize_profile_values(
    values: NDArray[np.float32] | NDArray[np.uint8],
    *,
    edge_frac: float = 0.12,
    inner_frac: float = 0.18,
    threshold: float = 10.0,
) -> OutlineCheck:
    """Summarize bilateral edge emphasis from grayscale or RGB samples."""
    brightness = profile_brightness(values)
    n = len(brightness)
    edge_n = max(2, int(round(n * edge_frac)))
    inner_n = max(2, int(round(n * inner_frac)))

    left_edge = float(brightness[:edge_n].mean())
    left_inner = float(brightness[edge_n : edge_n + inner_n].mean())
    right_edge = float(brightness[-edge_n:].mean())
    right_inner = float(brightness[-edge_n - inner_n : -edge_n].mean())

    left_delta = left_edge - left_inner
    right_delta = right_edge - right_inner
    return OutlineCheck(
        left_edge_delta=left_delta,
        right_edge_delta=right_delta,
        has_dark_outline=left_delta < -threshold and right_delta < -threshold,
        has_bright_outline=left_delta > threshold and right_delta > threshold,
    )


def summarize_rendered_ship_profile(
    rgba: NDArray[np.uint8],
    *,
    bg_color: tuple[int, int, int] = (40, 60, 90),
    sample_count: int = 128,
) -> BeamProfileMetrics:
    """Measure a rendered ship without regenerating it from SVG."""
    positions, brightness, x0_px, y_px, x1_px, row_count = sample_midship_beam_profile(
        rgba,
        bg_color=bg_color,
        sample_count=sample_count,
    )
    return analyze_beam_profile(
        positions,
        brightness,
        x0_px=x0_px,
        x1_px=x1_px,
        y_px=y_px,
        row_count=row_count,
    )


def analyze_beam_profile(
    positions: NDArray[np.float32],
    brightness: NDArray[np.float32],
    *,
    x0_px: int = 0,
    x1_px: int = 0,
    y_px: int = 0,
    row_count: int = 0,
    edge_frac: float = 0.12,
    inner_frac: float = 0.18,
    threshold: float = 10.0,
) -> BeamProfileMetrics:
    """Classify whether a profile exhibits a bilateral edge outline."""
    summary = summarize_profile_values(
        brightness,
        edge_frac=edge_frac,
        inner_frac=inner_frac,
        threshold=threshold,
    )

    return BeamProfileMetrics(
        positions=positions,
        brightness=brightness,
        left_edge_delta=summary.left_edge_delta,
        right_edge_delta=summary.right_edge_delta,
        has_dark_outline=summary.has_dark_outline,
        has_bright_outline=summary.has_bright_outline,
        x0_px=x0_px,
        x1_px=x1_px,
        y_px=y_px,
        row_count=row_count,
    )


def render_ship_profile_metrics(
    ship_class: str,
    *,
    seed: int,
    beam_px: int = 96,
    length_px: int = 480,
    bg_color: tuple[int, int, int] = (40, 60, 90),
    hull_noise: float = 0.005,
    deck_scatter_density: float = 3.0,
    sample_count: int = 128,
) -> BeamProfileMetrics:
    """Render one ship and summarize its beam-direction brightness profile."""
    svg = generate_ship_svg(
        ship_class,
        rng=random.Random(seed),
        hull_noise=hull_noise,
        deck_scatter_density=deck_scatter_density,
    )
    rgba = rasterize_ship_svg(svg, beam_px, length_px)
    return summarize_rendered_ship_profile(
        rgba,
        bg_color=bg_color,
        sample_count=sample_count,
    )

