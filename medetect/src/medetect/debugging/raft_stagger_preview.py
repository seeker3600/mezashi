"""Visual QA script for raft_tight stagger feature.

Generates a grid of tiles that contain only raft_tight clusters so the
diagonal/staggered arrangement can be inspected visually.

Usage:
    pixi run python -m medetect.debugging raft-stagger-preview
    pixi run python -m medetect.debugging raft-stagger-preview --output-dir debug_runs/stagger-preview --count 32
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _render_raft_tight_tile(
    seed: int,
    image_size: int = 256,
    n_ships_range: tuple[int, int] = (3, 6),
    resolution_m: float = 3.0,
    length_range: tuple[float, float] = (20.0, 80.0),
    bg_color: tuple[int, int, int] = (50, 80, 60),
) -> tuple[np.ndarray, bool]:
    """Render a single tile with exactly one raft_tight cluster.

    Returns (rgb_image, stagger_was_active).
    """
    from medetect.datagen.placement import _place_cluster

    rng = random.Random(seed)

    water_mask = np.ones((image_size, image_size), dtype=bool)
    occupancy = np.zeros((image_size, image_size), dtype=bool)
    background = np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)

    # Force raft_tight layout by patching choices
    original_choices = rng.choices

    stagger_state: list[bool] = []

    def _patched_choices(population, weights=None, *, cum_weights=None, k=1):
        pop_list = list(population)
        if set(pop_list) == {"raft_tight", "raft_open", "area_scattered"}:
            return ["raft_tight"]
        return original_choices(population, weights=weights, cum_weights=cum_weights, k=k)

    rng.choices = _patched_choices  # type: ignore[method-assign]

    _place_cluster(
        water_mask=water_mask,
        occupancy=occupancy,
        svg_metas=None,
        resolution_m=resolution_m,
        rng=rng,
        cluster_size_range=n_ships_range,
        blur_sigma=0.5,
        alpha_range=(0.85, 0.95),
        class_id=0,
        image_size=image_size,
        background=background,
        length_range=length_range,
    )

    return background, False


def render_raft_stagger_grid(
    output_dir: Path,
    count: int = 32,
    *,
    seed_offset: int = 0,
    cols: int = 8,
    image_size: int = 256,
    resolution_m: float = 3.0,
    length_range: tuple[float, float] = (20.0, 80.0),
    n_ships_range: tuple[int, int] = (3, 6),
) -> None:
    """Generate a grid of raft_tight cluster tiles and save as a mosaic PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[np.ndarray] = []
    for i in range(count):
        seed = seed_offset + i
        tile, _ = _render_raft_tight_tile(
            seed=seed,
            image_size=image_size,
            n_ships_range=n_ships_range,
            resolution_m=resolution_m,
            length_range=length_range,
        )
        tiles.append(tile)

    rows = math.ceil(count / cols)
    pad = 4
    canvas_w = cols * image_size + (cols - 1) * pad
    canvas_h = rows * image_size + (rows - 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        col = idx % cols
        row = idx // cols
        x0 = col * (image_size + pad)
        y0 = row * (image_size + pad)
        canvas[y0 : y0 + image_size, x0 : x0 + image_size] = tile

    out_path = output_dir / "raft_stagger_preview.png"
    Image.fromarray(canvas).save(out_path)
    print(f"Saved {count} tiles -> {out_path}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="raft_tight stagger visual QA")
    parser.add_argument("--output-dir", type=Path, default=Path("debug_runs/stagger-preview"))
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resolution-m", type=float, default=3.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args(argv)

    render_raft_stagger_grid(
        output_dir=args.output_dir,
        count=args.count,
        cols=args.cols,
        image_size=args.image_size,
        resolution_m=args.resolution_m,
        seed_offset=args.seed_offset,
    )
