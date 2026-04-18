"""Single-tile composition and compatibility exports for datagen."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from PIL import Image
from rasterio.windows import Window

import medetect.datagen.pipeline as _pipeline

from medetect.datagen.obb import compute_obb_corners, format_obb_label
from medetect.datagen.placement import (
    _RaftShipPlacement,
    _geometry_projection_extents,
    _place_area_cluster,
    _place_cluster,
    _render_vector_raft_cluster,
    _stamp_occupancy,
    find_water_position,
)
from medetect.datagen.scene import (
    SingleShipPlacement,
    _blend_rgba_layer,
    _cluster_scene_origin,
    _composite_rgba,
    _make_shadow_rgba,
    _downsample_cluster_layer,
    _rasterize_ship_scene,
    _render_ship,
    _sample_shadow_alpha,
    _sample_water_tint,
    _shadow_alpha_for_ship,
    _shadow_blur_sigma,
    _shadow_offset_pixels,
    blend_shadow,
    blend_ship,
)
from medetect.datagen.ship import _SvgMeta, _pick_svg, _ship_class_id
from medetect.datagen.wake import pick_motion_state, render_wake
from medetect.datagen.water_mask import (
    CoastlineIndex,
    erode_mask,
    make_water_mask_from_coastline,
    make_water_mask_from_rgb,
    make_water_mask_from_scl,
)

generate_dataset = _pipeline.generate_dataset
generate_false_negatives = _pipeline.generate_false_negatives
_false_source_grid = _pipeline._false_source_grid
_worker_init = _pipeline._worker_init
_run_compose_task = _pipeline._run_compose_task
_write_dataset_yaml = _pipeline._write_dataset_yaml

logger = logging.getLogger(__name__)

_DARK_TILE_THRESHOLD: float = 10.0

__all__ = [
    "SingleShipPlacement",
    "_RaftShipPlacement",
    "_blend_rgba_layer",
    "_cluster_scene_origin",
    "_compose_one",
    "_composite_rgba",
    "_downsample_cluster_layer",
    "_false_source_grid",
    "_geometry_projection_extents",
    "_place_area_cluster",
    "_place_cluster",
    "_rasterize_ship_scene",
    "_read_scl_tile",
    "_read_tile",
    "_render_ship",
    "_render_vector_raft_cluster",
    "_run_compose_task",
    "_sample_water_tint",
    "_scl_path_for",
    "_stamp_occupancy",
    "_worker_init",
    "_write_dataset_yaml",
    "augment_tile",
    "blend_ship",
    "find_water_position",
    "generate_dataset",
    "generate_false_negatives",
    "is_dark_tile",
    "make_nodata_mask",
]


def __getattr__(name: str):
    if name in {"_worker_svg_metas", "_worker_coastline_index"}:
        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def augment_tile(
    tile: NDArray[np.uint8],
    rng: random.Random,
) -> NDArray[np.uint8]:
    """Apply random colour augmentation to a background tile."""
    img = tile.astype(np.float32)

    for channel in range(3):
        img[:, :, channel] *= rng.uniform(0.85, 1.15)

    gamma = rng.uniform(0.8, 1.2)
    img = np.clip(img, 0, 255)
    img = 255.0 * (img / 255.0) ** gamma

    img += rng.uniform(-10, 10)

    return np.clip(img, 0, 255).astype(np.uint8)


def is_dark_tile(tile: NDArray[np.uint8], threshold: float = _DARK_TILE_THRESHOLD) -> bool:
    """Return True when the tile is a satellite blackout / no-data area."""
    return float(tile.mean()) < threshold


def make_nodata_mask(tile: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Return ``True`` for pixels that are pure black (#000000)."""
    return (tile[:, :, 0] == 0) & (tile[:, :, 1] == 0) & (tile[:, :, 2] == 0)


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
        scale = scl_src.res[0] / 10.0
        scl_col = int(col / scale)
        scl_row = int(row / scale)
        scl_size = max(1, int(size / scale))

        scl_col = min(scl_col, scl_src.width - scl_size)
        scl_row = min(scl_row, scl_src.height - scl_size)
        scl_col = max(0, scl_col)
        scl_row = max(0, scl_row)

        window = Window(scl_col, scl_row, scl_size, scl_size)
        scl = scl_src.read(1, window=window)

    img = Image.fromarray(scl)
    img = img.resize((target_size, target_size), Image.NEAREST)
    return np.array(img)


def _scl_path_for(visual_path: Path) -> Path:
    """Derive SCL file path from a visual TIF path."""
    name = visual_path.name
    scl_name = name.replace("_visual.tif", "_SCL_20m.tif")
    return visual_path.parent / scl_name


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
    debug_bg_color: tuple[int, int, int] | None = None,
    shadow_alpha_scale: float = 1.0,
    shadow_length_range: tuple[float, float] = (0.0, 3.75),
    shadow_azimuth_rad: float | None = None,
    coastline_index: CoastlineIndex | None = None,
) -> tuple[NDArray[np.uint8], list[str], int] | None:
    """Compose one training image. Returns ``(tile, labels, n_clusters)``."""
    with rasterio.open(tif_path) as src:
        if geo_scale is not None:
            src_tile = max(1, round(image_size * geo_scale))
            ship_resolution = resolution if resolution is not None else 10.0
        else:
            native_res = (src.res[0] + src.res[1]) / 2.0

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
                    tif_path.name,
                    col,
                    row,
                )
                continue

            if src_tile != image_size:
                img = Image.fromarray(tile)
                img = img.resize((image_size, image_size), Image.BILINEAR)
                tile = np.array(img)

            if is_dark_tile(tile):
                logger.debug(
                    "Dark tile in %s at col=%d row=%d (mean=%.1f) — retrying",
                    tif_path.name,
                    col,
                    row,
                    float(tile.mean()),
                )
                continue

            scl_file = _scl_path_for(tif_path)
            scl = _read_scl_tile(scl_file, col, row, src_tile, image_size)
            if scl is not None:
                water_mask = make_water_mask_from_scl(scl)
            else:
                water_mask = make_water_mask_from_rgb(tile)

            water_mask &= ~make_nodata_mask(tile)

            if coastline_index is not None:
                window = Window(col, row, src_tile, src_tile)
                tile_transform = src.window_transform(window)
                if src_tile != image_size:
                    bounds = rasterio.transform.array_bounds(
                        src_tile,
                        src_tile,
                        tile_transform,
                    )
                    tile_transform = rasterio.transform.from_bounds(
                        *bounds,
                        image_size,
                        image_size,
                    )
                tile_bounds = rasterio.transform.array_bounds(
                    image_size,
                    image_size,
                    tile_transform,
                )
                coastline_geoms = coastline_index.query(tile_bounds)
                coastline_mask = make_water_mask_from_coastline(
                    coastline_geoms,
                    tile,
                    tile_transform,
                    image_size,
                    image_size,
                )
                water_mask &= coastline_mask

            water_mask = erode_mask(water_mask, erode_coast)

            water_ratio = water_mask.sum() / water_mask.size
            if water_ratio >= min_water_ratio:
                break
        else:
            try:
                return tile, [], 0  # type: ignore[possibly-undefined]
            except NameError:
                return None

    if debug_bg_color is None:
        tile = augment_tile(tile, rng)
    else:
        tile = np.zeros_like(tile)
        tile[:, :] = debug_bg_color

    n_events = rng.randint(*ships_per_image)
    occupancy = np.zeros((image_size, image_size), dtype=bool)
    labels: list[str] = []
    n_clusters = 0
    single_ships: list[SingleShipPlacement] = []
    tile_shadow_azimuth = shadow_azimuth_rad if shadow_azimuth_rad is not None else rng.uniform(0.0, 2.0 * math.pi)
    shadow_length_min, shadow_length_max = shadow_length_range
    shadow_length_min, shadow_length_max = sorted((shadow_length_min, shadow_length_max))
    tile_shadow_length = rng.uniform(shadow_length_min, shadow_length_max)
    tile_shadow_alpha = _sample_shadow_alpha(rng) if shadow_alpha_scale > 0.0 else 0.0

    for _ in range(n_events):
        is_cluster = rng.random() < cluster_prob

        if is_cluster:
            new_labels = _place_cluster(
                water_mask,
                occupancy,
                svg_metas,
                ship_resolution,
                rng,
                cluster_size,
                ship_blur_sigma,
                ship_alpha,
                class_id,
                image_size,
                tile,
                ship_length_range,
                length_exponent,
                size_threshold,
                mixed_prob=cluster_mixed_prob,
                shadow_azimuth_rad=tile_shadow_azimuth,
                shadow_length=tile_shadow_length,
                shadow_alpha=tile_shadow_alpha,
                shadow_alpha_scale=shadow_alpha_scale,
            )
            labels.extend(new_labels)
            if new_labels:
                n_clusters += 1
        else:
            svg_text = _pick_svg(svg_metas, rng, ship_length_range)
            angle_deg = rng.uniform(0, 360)
            angle_rad = math.radians(angle_deg)
            rotated, _cls_name, bw, lh, _lb = _render_ship(
                svg_text,
                ship_resolution,
                rng,
                ship_blur_sigma,
                ship_length_range,
                angle_deg=angle_deg,
                length_exponent=length_exponent,
                supersample=4,
            )

            available = water_mask & ~occupancy
            pos = find_water_position(available, bw, lh, angle_rad, rng)
            if pos is None:
                continue

            cx, cy = pos
            alpha = rng.uniform(*ship_alpha)
            water_tint = _sample_water_tint(tile, cx, cy)
            ship_state = pick_motion_state(rng)
            shadow_rgba = None
            if shadow_alpha_scale > 0.0 and tile_shadow_alpha > 0.0:
                offset_x, offset_y = _shadow_offset_pixels(
                    bw,
                    lh,
                    tile_shadow_azimuth,
                    tile_shadow_length,
                )
                cast_length = math.hypot(offset_x, offset_y)
                shadow_rgba = _make_shadow_rgba(
                    rotated,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    blur_sigma=_shadow_blur_sigma(bw, lh, cast_length),
                    alpha_scale=_shadow_alpha_for_ship(bw, lh),
                )
            _stamp_occupancy(occupancy, cx, cy, bw, lh, angle_rad)
            corners = compute_obb_corners(
                float(cx),
                float(cy),
                float(bw),
                float(lh),
                angle_rad,
            )
            cid = _ship_class_id(lh, ship_resolution, class_id, size_threshold)
            single_ships.append(
                SingleShipPlacement(
                    cx=cx,
                    cy=cy,
                    rotated=rotated,
                    bw=bw,
                    lh=lh,
                    angle_rad=angle_rad,
                    alpha=alpha,
                    water_tint=water_tint,
                    shadow_rgba=shadow_rgba,
                    ship_state=ship_state,
                    class_id=cid,
                    corners=corners,
                )
            )

    for ship in single_ships:
        render_wake(
            tile,
            water_mask,
            float(ship.cx),
            float(ship.cy),
            ship.bw,
            ship.lh,
            ship.angle_rad,
            ship.ship_state,
            rng,
            wake_prob_scale=wake_prob_scale,
            wake_alpha_scale=wake_alpha_scale,
        )

    for ship in single_ships:
        if ship.shadow_rgba is not None:
            blend_shadow(
                tile,
                ship.shadow_rgba,
                ship.cx,
                ship.cy,
                alpha_factor=tile_shadow_alpha * shadow_alpha_scale,
                clip_mask=water_mask,
            )

    for ship in single_ships:
        blend_ship(tile, ship.rotated, ship.cx, ship.cy, ship.alpha, ship.water_tint)
        labels.append(format_obb_label(ship.class_id, ship.corners, image_size, image_size))

    return tile, labels, n_clusters