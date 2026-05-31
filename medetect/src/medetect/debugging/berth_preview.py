from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

import medetect.datagen.placement as placement_mod
from medetect.datagen.placement import (
    _build_berth_runs,
    _ordered_berth_mode_attempts,
    _place_berthed_cluster,
    _place_cluster,
)

_IMAGE_SIZE = 256
_WATER_RGB = np.array([36, 74, 108], dtype=np.uint8)
_LAND_RGB = np.array([110, 104, 88], dtype=np.uint8)


def _make_background(water_mask: np.ndarray) -> np.ndarray:
    background = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)
    background[:, :] = _LAND_RGB
    background[water_mask] = _WATER_RGB
    return background


def _curved_shore_mask(
    shore_points: list[tuple[float, float]],
) -> np.ndarray:
    water_mask = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=bool)
    shore_x = np.full(_IMAGE_SIZE, shore_points[0][0], dtype=float)
    for (x0, y0), (x1, y1) in zip(shore_points, shore_points[1:]):
        y_start = max(0, int(math.floor(min(y0, y1))))
        y_stop = min(_IMAGE_SIZE - 1, int(math.ceil(max(y0, y1))))
        for y in range(y_start, y_stop + 1):
            if y1 == y0:
                shore_x[y] = x1
                continue
            t = (y - y0) / (y1 - y0)
            shore_x[y] = x0 + (x1 - x0) * t
    shore_x[: int(shore_points[0][1])] = shore_points[0][0]
    shore_x[int(shore_points[-1][1]) :] = shore_points[-1][0]
    for y, x in enumerate(shore_x):
        water_mask[y, max(0, int(math.ceil(x))) :] = True
    return water_mask


def _write_case(
    output_dir: Path,
    name: str,
    water_mask: np.ndarray,
    berth_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    berth_stern_prob: float,
    seed: int,
) -> Path:
    background = _make_background(water_mask)

    labels = _place_cluster(
        water_mask,
        np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=bool),
        None,
        resolution_m=3.0,
        rng=random.Random(seed),
        cluster_size_range=(3, 3),
        blur_sigma=0.2,
        alpha_range=(0.9, 0.95),
        class_id=0,
        image_size=_IMAGE_SIZE,
        background=background,
        length_range=(20.0, 80.0),
        mixed_prob=0.0,
        berth_prob=1.0,
        berth_stern_prob=berth_stern_prob,
        berth_water_mask=water_mask,
        berth_segments=berth_segments,
        shadow_azimuth_rad=math.pi / 5.0,
        shadow_length=1.5,
        shadow_alpha=0.08,
        shadow_alpha_scale=1.0,
    )
    if not labels:
        msg = f"Failed to render berth preview: {name}"
        raise RuntimeError(msg)

    image_path = output_dir / f"{name}.png"
    label_path = output_dir / f"{name}.txt"
    Image.fromarray(background).save(image_path)
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return image_path


def _classify_single_seed(
    water_mask: np.ndarray,
    berth_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    seed: int,
) -> str:
    runs = _build_berth_runs(berth_segments, water_mask)
    rng = random.Random(seed)
    berth_modes = _ordered_berth_mode_attempts(
        runs,
        3.0,
        (20.0, 80.0),
        rng,
        berth_stern_prob=0.5,
        ship_gap=4.0,
        min_required_ships=1,
    )

    for berth_stern in berth_modes:
        labels = _place_berthed_cluster(
            water_mask,
            np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=bool),
            berth_segments,
            None,
            resolution_m=3.0,
            rng=rng,
            n_ships=1,
            alpha_range=(0.9, 0.95),
            class_id=0,
            image_size=_IMAGE_SIZE,
            background=_make_background(water_mask),
            length_range=(20.0, 80.0),
            length_exponent=1.0,
            size_thresholds=None,
            mixed=False,
            berth_stern=berth_stern,
            blur_sigma=0.2,
            shadow_azimuth_rad=math.pi / 5.0,
            shadow_length=1.5,
            shadow_alpha=0.08,
            shadow_alpha_scale=1.0,
        )
        if labels:
            return "single_stern" if berth_stern else "single_alongside"

    return "fallback_open"


def _classify_cluster_seed(
    water_mask: np.ndarray,
    berth_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    seed: int,
) -> str:
    category: str | None = None
    original_place_coastal = placement_mod._place_coastal_raft_cluster

    def _capture_coastal(*args, **kwargs):
        nonlocal category
        labels = original_place_coastal(*args, **kwargs)
        if labels:
            category = "cluster_tight_stern" if kwargs["berth_stern"] else "cluster_tight_alongside"
        return labels

    placement_mod._place_coastal_raft_cluster = _capture_coastal
    try:
        _place_cluster(
            water_mask,
            np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=3.0,
            rng=random.Random(seed),
            cluster_size_range=(3, 3),
            blur_sigma=0.2,
            alpha_range=(0.9, 0.95),
            class_id=0,
            image_size=_IMAGE_SIZE,
            background=_make_background(water_mask),
            length_range=(20.0, 80.0),
            mixed_prob=0.0,
            berth_prob=1.0,
            berth_stern_prob=0.5,
            coastal_raft_tight_prob=1.0,
            coastal_raft_min_ships=2,
            berth_water_mask=water_mask,
            berth_segments=berth_segments,
            shadow_azimuth_rad=math.pi / 5.0,
            shadow_length=1.5,
            shadow_alpha=0.08,
            shadow_alpha_scale=1.0,
        )
    finally:
        placement_mod._place_coastal_raft_cluster = original_place_coastal

    return category or "fallback_open"


def count_berth_seed_sweep(seed_count: int = 64) -> dict[str, int]:
    """Count shoreline routing outcomes across a deterministic seed sweep."""
    vertical_water = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=bool)
    vertical_water[:, 96:] = True
    berth_segments = [((96.0, 24.0), (96.0, 232.0))]
    counts = {
        "single_alongside": 0,
        "single_stern": 0,
        "cluster_tight_alongside": 0,
        "cluster_tight_stern": 0,
        "fallback_open": 0,
    }

    for seed in range(seed_count):
        counts[_classify_single_seed(vertical_water, berth_segments, seed=seed)] += 1
        counts[_classify_cluster_seed(vertical_water, berth_segments, seed=seed)] += 1

    return counts


def render_berth_previews(
    output_dir: Path = Path("debug_runs/berth-implementation-qa"),
) -> dict[str, Path]:
    """Render deterministic berth QA preview images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    vertical_water = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=bool)
    vertical_water[:, 96:] = True
    horizontal_water = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE), dtype=bool)
    horizontal_water[96:, :] = True
    curved_points = [(96.0, 24.0), (106.0, 128.0), (120.0, 232.0)]
    curved_water = _curved_shore_mask(curved_points)
    curved_segments = list(zip(curved_points, curved_points[1:]))

    outputs = {
        "alongside_cluster": _write_case(
            output_dir,
            "alongside_cluster",
            vertical_water,
            [((96.0, 24.0), (96.0, 232.0))],
            berth_stern_prob=0.0,
            seed=7,
        ),
        "stern_to_cluster": _write_case(
            output_dir,
            "stern_to_cluster",
            horizontal_water,
            [((24.0, 96.0), (232.0, 96.0))],
            berth_stern_prob=1.0,
            seed=11,
        ),
        "curved_alongside_cluster": _write_case(
            output_dir,
            "curved_alongside_cluster",
            curved_water,
            curved_segments,
            berth_stern_prob=0.0,
            seed=13,
        ),
        "curved_stern_to_cluster": _write_case(
            output_dir,
            "curved_stern_to_cluster",
            curved_water,
            curved_segments,
            berth_stern_prob=1.0,
            seed=17,
        ),
    }
    counts = count_berth_seed_sweep()
    counts_path = output_dir / "seed_sweep_counts.txt"
    counts_path.write_text(
        "\n".join(f"{name}: {value}" for name, value in counts.items()) + "\n",
        encoding="utf-8",
    )
    outputs["seed_sweep_counts"] = counts_path
    return outputs