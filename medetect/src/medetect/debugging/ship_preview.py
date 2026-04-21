"""Ship preview renderers for manual visual QA."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from medetect.debugging.ship_profile import composite_rgba_on_background
from medetect.datagen.render import rasterize_ship_svg
from medetect.shipgen.gen import generate_ship_svg

DEFAULT_PREVIEW_CLASSES = (
    "destroyer",
    "frigate",
    "carrier",
    "fishing_trawler",
    "tug_harbor",
    "barge",
)


def render_ship_grid(
    classes: Sequence[str],
    *,
    seed: int = 42,
    beam_px: int = 60,
    length_px: int = 400,
    bg_color: tuple[int, int, int] = (40, 60, 90),
) -> np.ndarray:
    """Render a single-row preview grid for ship classes."""
    images = []
    for ship_class in classes:
        svg = generate_ship_svg(ship_class, rng=random.Random(seed))
        rgba = rasterize_ship_svg(svg, beam_px, length_px)
        images.append(composite_rgba_on_background(rgba, bg_color=bg_color))

    pad = 10
    max_h = max(image.shape[0] for image in images)
    total_w = sum(image.shape[1] for image in images) + pad * (len(images) - 1)
    canvas = np.zeros((max_h, total_w, 3), dtype=np.uint8)
    canvas[:, :] = bg_color

    x = 0
    for image in images:
        h, w = image.shape[:2]
        y_off = (max_h - h) // 2
        canvas[y_off : y_off + h, x : x + w] = image
        x += w + pad
    return canvas


def render_multi_seed_grid(
    classes: Sequence[str],
    seeds: Sequence[int],
    *,
    beam_px: int = 50,
    length_px: int = 300,
    bg_color: tuple[int, int, int] = (40, 60, 90),
) -> np.ndarray:
    """Render a grid whose rows are seeds and columns are ship classes."""
    pad = 6
    all_images: list[list[np.ndarray]] = []
    for seed in seeds:
        row = []
        for ship_class in classes:
            svg = generate_ship_svg(ship_class, rng=random.Random(seed))
            rgba = rasterize_ship_svg(svg, beam_px, length_px)
            row.append(composite_rgba_on_background(rgba, bg_color=bg_color))
        all_images.append(row)

    n_rows = len(seeds)
    n_cols = len(classes)
    max_h = max(image.shape[0] for row in all_images for image in row)
    max_w = max(image.shape[1] for row in all_images for image in row)
    canvas_w = n_cols * (max_w + pad) - pad
    canvas_h = n_rows * (max_h + pad) - pad
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = bg_color

    for row_index, row in enumerate(all_images):
        for col_index, image in enumerate(row):
            h, w = image.shape[:2]
            x0 = col_index * (max_w + pad) + (max_w - w) // 2
            y0 = row_index * (max_h + pad) + (max_h - h) // 2
            canvas[y0 : y0 + h, x0 : x0 + w] = image
    return canvas


def save_ship_previews(
    output_dir: Path,
    *,
    classes: Sequence[str] = DEFAULT_PREVIEW_CLASSES,
    seeds: Sequence[int] = (42, 7, 123, 999),
    bg_color: tuple[int, int, int] = (40, 60, 90),
) -> dict[str, Path]:
    """Write default ship preview assets and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "ship_preview.png"
    multi_path = output_dir / "ship_multi_preview.png"

    preview_seed = seeds[0] if seeds else 42
    Image.fromarray(render_ship_grid(classes, seed=preview_seed, bg_color=bg_color)).save(preview_path)
    Image.fromarray(render_multi_seed_grid(classes, seeds, bg_color=bg_color)).save(multi_path)
    return {"ship_preview": preview_path, "ship_multi_preview": multi_path}
