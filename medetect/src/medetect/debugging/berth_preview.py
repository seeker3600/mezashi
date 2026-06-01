from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from medetect.datagen.placement import _place_cluster

_IMAGE_SIZE = 256
_WATER_RGB = np.array([36, 74, 108], dtype=np.uint8)
_LAND_RGB = np.array([110, 104, 88], dtype=np.uint8)


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
    background = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)
    background[:, :] = _LAND_RGB
    background[water_mask] = _WATER_RGB

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
    return outputs