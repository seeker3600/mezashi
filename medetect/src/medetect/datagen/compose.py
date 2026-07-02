"""Single-tile composition and compatibility exports for datagen."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import random
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from PIL import Image
from rasterio.crs import CRS
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window

import medetect.datagen.pipeline as _pipeline

from medetect.datagen.obb import compute_obb_corners, format_obb_label
from medetect.datagen.placement import (
    _RaftShipPlacement,
    _build_berth_runs,
    _geometry_projection_extents,
    _place_berthed_cluster,
    _place_cluster,
    _render_vector_raft_cluster,
    _stamp_occupancy,
    find_water_position,
)
from medetect.datagen.scene import (
    DEFAULT_EDGE_HARDNESS,
    SingleShipPlacement,
    _blend_rgba_layer,
    _cluster_scene_origin,
    _composite_rgba,
    _make_shadow_rgba,
    _downsample_cluster_layer,
    _edge_hardness_to_blur_sigma,
    _rasterize_ship_scene,
    _render_ship,
    _render_ship_from_dimensions,
    _sample_shadow_alpha,
    _sample_water_tint,
    _shadow_alpha_for_ship,
    _shadow_blur_sigma,
    _shadow_offset_pixels,
    blend_shadow,
    blend_ship,
)
from medetect.datagen.ship import (
    MIN_SHIP_BEAM_PX,
    MIN_SHIP_LENGTH_PX,
    SHIP_LENGTHS_M,
    _DEFAULT_LENGTH_M,
    _SvgMeta,
    _pick_svg,
    _resolve_ship_dimensions,
    _ship_class_id,
)
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
_COASTLINE_CRS = CRS.from_epsg(4326)
_PLACEMENT_FAILURE_CHECK_INTERVAL = 8
OPEN_WATER_MIN_RATIO: float = 0.75
MIXED_MIN_RATIO: float = 0.35
_ROTATION_SOURCE_SCALE: float = math.sqrt(2.0)

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


def _iter_line_geometries(geometry):
    """Yield linear components from a Shapely geometry."""
    if geometry.is_empty:
        return
    geom_type = getattr(geometry, "geom_type", "")
    if geom_type in {"LineString", "LinearRing"}:
        yield geometry
        return
    if geom_type == "Polygon":
        yield geometry.exterior
        for interior in geometry.interiors:
            yield interior
        return
    for child in getattr(geometry, "geoms", ()):  # MultiLineString / GeometryCollection
        yield from _iter_line_geometries(child)


def _coastline_to_pixel_segments(
    coastline_geoms: list,
    transform: object,
    width: int,
    height: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Project coastline geometries into image-space line segments."""
    inv_transform = ~transform
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for geometry in coastline_geoms:
        for line in _iter_line_geometries(geometry):
            coords = list(getattr(line, "coords", ()))
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                px0, py0 = inv_transform * (x0, y0)
                px1, py1 = inv_transform * (x1, y1)
                if (
                    max(px0, px1) < -1.0
                    or max(py0, py1) < -1.0
                    or min(px0, px1) > width + 1.0
                    or min(py0, py1) > height + 1.0
                ):
                    continue
                segments.append(((float(px0), float(py0)), (float(px1), float(py1))))
    return segments


def _reproject_coastline_geometries(
    coastline_geoms: list,
    src_crs: object,
    dst_crs: object,
) -> list:
    """Reproject Shapely coastline geometries between CRS definitions."""
    if not coastline_geoms:
        return []

    src_crs_obj = CRS.from_user_input(src_crs)
    dst_crs_obj = CRS.from_user_input(dst_crs)
    if src_crs_obj == dst_crs_obj:
        return list(coastline_geoms)

    from shapely.geometry import mapping, shape

    reprojected = []
    for geometry in coastline_geoms:
        geom_json = transform_geom(src_crs_obj, dst_crs_obj, mapping(geometry))
        transformed = shape(geom_json)
        if not transformed.is_empty:
            reprojected.append(transformed)
    return reprojected


def _minimum_ship_bbox(
    ship_resolution: float,
    ship_length_range: tuple[float, float] | None,
) -> tuple[int, int]:
    min_length_m = min(
        _DEFAULT_LENGTH_M[0],
        *(lo for lo, _hi in SHIP_LENGTHS_M.values()),
    )
    if ship_length_range is not None:
        min_length_m = ship_length_range[0]
    min_length_px = max(MIN_SHIP_LENGTH_PX, round(min_length_m / ship_resolution))
    return MIN_SHIP_BEAM_PX + 2, min_length_px + 2


def _has_full_open_box(
    mask: NDArray[np.bool_],
    box_w: int,
    box_h: int,
) -> bool:
    if box_w <= 0 or box_h <= 0:
        return False

    mask_h, mask_w = mask.shape
    if box_w > mask_w or box_h > mask_h:
        return False

    integral = np.pad(
        mask.astype(np.int32),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    ).cumsum(axis=0).cumsum(axis=1)
    window_sums = (
        integral[box_h:, box_w:]
        - integral[:-box_h, box_w:]
        - integral[box_h:, :-box_w]
        + integral[:-box_h, :-box_w]
    )
    return bool(np.any(window_sums == box_w * box_h))


def _wake_occlusion_mask(
    occupancy: NDArray[np.bool_],
    ship: SingleShipPlacement,
) -> NDArray[np.bool_]:
    """Return occupied pixels excluding the wake source ship footprint."""
    source_mask = np.zeros_like(occupancy)
    _stamp_occupancy(
        source_mask,
        ship.cx,
        ship.cy,
        ship.bw,
        ship.lh,
        ship.angle_rad,
    )
    return occupancy & ~source_mask


@dataclass
class _BackgroundContext:
    tile: NDArray[np.uint8]
    nodata_mask: NDArray[np.bool_]
    water_mask: NDArray[np.bool_]
    surface_water_mask: NDArray[np.bool_]
    surface_valid_mask: NDArray[np.bool_]
    berth_water_mask: NDArray[np.bool_] | None
    berth_segments: list[tuple[tuple[float, float], tuple[float, float]]]
    berth_runs: object | None


def _rotation_source_size(size: int) -> int:
    """Return a square size that can support arbitrary-angle center cropping."""
    return max(size, math.ceil(size * _ROTATION_SOURCE_SCALE))


def _center_crop_array(array: NDArray, output_size: int) -> NDArray:
    """Return the centered square crop with the requested output size."""
    height, width = array.shape[:2]
    top = max(0, (height - output_size) // 2)
    left = max(0, (width - output_size) // 2)
    return array[top : top + output_size, left : left + output_size].copy()


def _rotate_background_and_crop(
    tile: NDArray[np.uint8],
    angle_deg: float,
    output_size: int,
    *,
    resample: int,
) -> NDArray[np.uint8]:
    """Rotate a square background tile and return its centered crop."""
    rotated = Image.fromarray(tile).rotate(
        angle_deg,
        resample=resample,
        expand=False,
        fillcolor=0,
    )
    return _center_crop_array(np.asarray(rotated, dtype=tile.dtype), output_size)


def _rotate_mask_and_crop(
    mask: NDArray[np.bool_],
    angle_deg: float,
    output_size: int,
) -> NDArray[np.bool_]:
    """Rotate a boolean mask with nearest-neighbour resampling and center crop it."""
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    rotated = mask_img.rotate(
        angle_deg,
        resample=Image.NEAREST,
        expand=False,
        fillcolor=0,
    )
    cropped = _center_crop_array(np.asarray(rotated, dtype=np.uint8), output_size)
    return cropped > 0


def _ensure_mask_shape(
    mask: NDArray[np.bool_] | NDArray[np.uint8],
    output_size: int,
) -> NDArray[np.bool_]:
    """Resize arbitrary mask-like arrays to the requested square tile size."""
    if mask.shape == (output_size, output_size):
        return mask.astype(bool, copy=False)
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = mask_img.resize((output_size, output_size), Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _rotate_point_about_center(
    x: float,
    y: float,
    angle_rad: float,
    center: float,
) -> tuple[float, float]:
    """Rotate an image-space point around the square center."""
    dx = x - center
    dy = y - center
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    return (
        (cos_theta * dx) + (sin_theta * dy) + center,
        (-sin_theta * dx) + (cos_theta * dy) + center,
    )


def _rotate_segments_and_crop(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    angle_deg: float,
    source_size: int,
    output_size: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Rotate image-space line segments and translate them into the centered crop."""
    if not segments:
        return []

    angle_rad = math.radians(angle_deg)
    center = (source_size - 1) / 2.0
    crop_offset = (source_size - output_size) / 2.0
    rotated: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for (x0, y0), (x1, y1) in segments:
        rx0, ry0 = _rotate_point_about_center(x0, y0, angle_rad, center)
        rx1, ry1 = _rotate_point_about_center(x1, y1, angle_rad, center)
        rotated.append(
            (
                (rx0 - crop_offset, ry0 - crop_offset),
                (rx1 - crop_offset, ry1 - crop_offset),
            )
        )
    return rotated


def _extract_background_context(
    *,
    src: rasterio.DatasetReader,
    tif_path: Path,
    col: int,
    row: int,
    src_tile: int,
    image_size: int,
    coastline_index: CoastlineIndex | None,
) -> _BackgroundContext:
    """Read one background crop and derive all placement masks from it."""
    tile = _read_tile(src, col, row, src_tile)
    if src_tile != image_size:
        img = Image.fromarray(tile)
        img = img.resize((image_size, image_size), Image.BILINEAR)
        tile = np.asarray(img, dtype=np.uint8)

    nodata_mask = make_nodata_mask(tile)
    scl_file = _scl_path_for(tif_path)
    scl = _read_scl_tile(scl_file, col, row, src_tile, image_size)
    if scl is not None:
        water_mask = make_water_mask_from_scl(scl)
    else:
        water_mask = make_water_mask_from_rgb(tile)
    water_mask = _ensure_mask_shape(water_mask, image_size)

    water_mask &= ~nodata_mask
    surface_water_mask = water_mask.copy()
    surface_valid_mask = ~nodata_mask

    berth_water_mask: NDArray[np.bool_] | None = None
    berth_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    berth_runs = None

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
        query_bounds = tile_bounds
        if src.crs is not None and src.crs != _COASTLINE_CRS:
            query_bounds = transform_bounds(
                src.crs,
                _COASTLINE_CRS,
                *tile_bounds,
                densify_pts=21,
            )
        coastline_geoms = coastline_index.query(query_bounds)
        if coastline_geoms and src.crs is not None:
            coastline_geoms = _reproject_coastline_geometries(
                coastline_geoms,
                _COASTLINE_CRS,
                src.crs,
            )
        coastline_mask = make_water_mask_from_coastline(
            coastline_geoms,
            tile,
            tile_transform,
            image_size,
            image_size,
        )
        surface_water_mask = coastline_mask
        water_mask &= coastline_mask
        berth_water_mask = water_mask.copy()
        berth_segments = _coastline_to_pixel_segments(
            coastline_geoms,
            tile_transform,
            image_size,
            image_size,
        )
        if berth_segments:
            berth_runs = _build_berth_runs(berth_segments, berth_water_mask)

    return _BackgroundContext(
        tile=tile,
        nodata_mask=nodata_mask,
        water_mask=water_mask,
        surface_water_mask=surface_water_mask,
        surface_valid_mask=surface_valid_mask,
        berth_water_mask=berth_water_mask,
        berth_segments=berth_segments,
        berth_runs=berth_runs,
    )


def _rotate_background_context(
    context: _BackgroundContext,
    angle_deg: float,
    output_size: int,
) -> _BackgroundContext:
    """Rotate a prepared background context and crop it back to the final tile size."""
    rotated_tile = _rotate_background_and_crop(
        context.tile,
        angle_deg,
        output_size,
        resample=Image.BILINEAR,
    )
    rotated_nodata_mask = _rotate_mask_and_crop(
        context.nodata_mask,
        angle_deg,
        output_size,
    )
    rotated_surface_water_mask = _rotate_mask_and_crop(
        context.surface_water_mask,
        angle_deg,
        output_size,
    )
    rotated_water_mask = _rotate_mask_and_crop(
        context.water_mask,
        angle_deg,
        output_size,
    )
    rotated_surface_valid_mask = _rotate_mask_and_crop(
        context.surface_valid_mask,
        angle_deg,
        output_size,
    )
    rotated_berth_water_mask = None
    rotated_berth_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    rotated_berth_runs = None

    if context.berth_water_mask is not None:
        rotated_berth_water_mask = _rotate_mask_and_crop(
            context.berth_water_mask,
            angle_deg,
            output_size,
        )
    if context.berth_segments:
        rotated_berth_segments = _rotate_segments_and_crop(
            context.berth_segments,
            angle_deg,
            context.tile.shape[0],
            output_size,
        )
    if rotated_berth_water_mask is not None and rotated_berth_segments:
        rotated_berth_runs = _build_berth_runs(
            rotated_berth_segments,
            rotated_berth_water_mask,
        )

    return _BackgroundContext(
        tile=rotated_tile,
        nodata_mask=rotated_nodata_mask,
        water_mask=rotated_water_mask,
        surface_water_mask=rotated_surface_water_mask,
        surface_valid_mask=rotated_surface_valid_mask,
        berth_water_mask=rotated_berth_water_mask,
        berth_segments=rotated_berth_segments,
        berth_runs=rotated_berth_runs,
    )


def _expanded_window_for_rotation(
    *,
    col: int,
    row: int,
    base_size: int,
    expanded_size: int,
    max_width: int,
    max_height: int,
) -> tuple[int, int] | None:
    """Return a centered larger source window or None when it does not fit."""
    if expanded_size <= base_size:
        return col, row
    if expanded_size > max_width or expanded_size > max_height:
        return None

    center_col = col + (base_size / 2.0)
    center_row = row + (base_size / 2.0)
    expanded_col = round(center_col - (expanded_size / 2.0))
    expanded_row = round(center_row - (expanded_size / 2.0))
    expanded_col = min(max(0, expanded_col), max_width - expanded_size)
    expanded_row = min(max(0, expanded_row), max_height - expanded_size)
    return expanded_col, expanded_row


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


def classify_background_surface(water_ratio: float) -> str:
    """Classify tile background into sea-only / mixed / land-only buckets."""
    if water_ratio >= OPEN_WATER_MIN_RATIO:
        return "sea_only"
    if water_ratio <= MIXED_MIN_RATIO:
        return "land_only"
    return "mixed"


def _compose_one_with_surface_category(
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
    edge_hardness: float = DEFAULT_EDGE_HARDNESS,
    ship_alpha: tuple[float, float] = (0.7, 0.95),
    ship_length_range: tuple[float, float] | None = None,
    ship_lb_ratio_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
    berth_prob: float = 0.25,
    berth_stern_prob: float = 0.5,
    rng: random.Random,
    max_crop_attempts: int = 20,
    size_thresholds: tuple[float, ...] | None = None,
    wake_prob_scale: float = 1.0,
    wake_alpha_scale: float = 1.0,
    debug_bg_color: tuple[int, int, int] | None = None,
    shadow_alpha_scale: float = 1.0,
    shadow_length_range: tuple[float, float] = (0.0, 3.75),
    shadow_azimuth_rad: float | None = None,
    coastline_index: CoastlineIndex | None = None,
    offnadir_range: tuple[float, float] = (0.0, 0.0),
    shipgen_kwargs: dict | None = None,
    required_surface: str | None = None,
) -> tuple[NDArray[np.uint8], list[str], int, str] | None:
    """Compose one training image and include a background surface category."""
    result = _compose_one(
        tif_path=tif_path,
        svg_metas=svg_metas,
        image_size=image_size,
        resolution=resolution,
        geo_scale=geo_scale,
        ships_per_image=ships_per_image,
        cluster_prob=cluster_prob,
        cluster_size=cluster_size,
        cluster_mixed_prob=cluster_mixed_prob,
        class_id=class_id,
        erode_coast=erode_coast,
        min_water_ratio=min_water_ratio,
        edge_hardness=edge_hardness,
        ship_alpha=ship_alpha,
        ship_length_range=ship_length_range,
        ship_lb_ratio_range=ship_lb_ratio_range,
        length_exponent=length_exponent,
        berth_prob=berth_prob,
        berth_stern_prob=berth_stern_prob,
        rng=rng,
        max_crop_attempts=max_crop_attempts,
        size_thresholds=size_thresholds,
        wake_prob_scale=wake_prob_scale,
        wake_alpha_scale=wake_alpha_scale,
        debug_bg_color=debug_bg_color,
        shadow_alpha_scale=shadow_alpha_scale,
        shadow_length_range=shadow_length_range,
        shadow_azimuth_rad=shadow_azimuth_rad,
        coastline_index=coastline_index,
        offnadir_range=offnadir_range,
        shipgen_kwargs=shipgen_kwargs,
        include_surface_category=True,
        required_surface=required_surface,
    )
    return result


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
    edge_hardness: float = DEFAULT_EDGE_HARDNESS,
    ship_alpha: tuple[float, float] = (0.7, 0.95),
    ship_length_range: tuple[float, float] | None = None,
    ship_lb_ratio_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
    berth_prob: float = 0.25,
    berth_stern_prob: float = 0.5,
    rng: random.Random,
    max_crop_attempts: int = 20,
    size_thresholds: tuple[float, ...] | None = None,
    wake_prob_scale: float = 1.0,
    wake_alpha_scale: float = 1.0,
    debug_bg_color: tuple[int, int, int] | None = None,
    shadow_alpha_scale: float = 1.0,
    shadow_length_range: tuple[float, float] = (0.0, 3.75),
    shadow_azimuth_rad: float | None = None,
    coastline_index: CoastlineIndex | None = None,
    offnadir_range: tuple[float, float] = (0.0, 0.0),
    shipgen_kwargs: dict | None = None,
    include_surface_category: bool = False,
    required_surface: str | None = None,
) -> tuple[NDArray[np.uint8], list[str], int] | tuple[NDArray[np.uint8], list[str], int, str] | None:
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

        rotation_src_tile = _rotation_source_size(src_tile)
        rotation_image_size = _rotation_source_size(image_size)

        for _ in range(max_crop_attempts):
            if src.width <= src_tile or src.height <= src_tile:
                return None
            col = rng.randint(0, src.width - src_tile)
            row = rng.randint(0, src.height - src_tile)

            try:
                context = _extract_background_context(
                    src=src,
                    tif_path=tif_path,
                    col=col,
                    row=row,
                    src_tile=src_tile,
                    image_size=image_size,
                    coastline_index=coastline_index,
                )
            except rasterio.errors.RasterioIOError:
                logger.debug(
                    "Tile read error in %s at col=%d row=%d — retrying",
                    tif_path.name,
                    col,
                    row,
                )
                continue

            has_land = bool(
                np.any(context.surface_valid_mask & ~context.surface_water_mask)
            )
            if has_land:
                expanded_window = _expanded_window_for_rotation(
                    col=col,
                    row=row,
                    base_size=src_tile,
                    expanded_size=rotation_src_tile,
                    max_width=src.width,
                    max_height=src.height,
                )
                if expanded_window is None:
                    logger.debug(
                        "Land-containing tile in %s at col=%d row=%d cannot fit rotation source window — retrying",
                        tif_path.name,
                        col,
                        row,
                    )
                    continue
                expanded_col, expanded_row = expanded_window
                try:
                    expanded_context = _extract_background_context(
                        src=src,
                        tif_path=tif_path,
                        col=expanded_col,
                        row=expanded_row,
                        src_tile=rotation_src_tile,
                        image_size=rotation_image_size,
                        coastline_index=coastline_index,
                    )
                except rasterio.errors.RasterioIOError:
                    logger.debug(
                        "Expanded tile read error in %s at col=%d row=%d — retrying",
                        tif_path.name,
                        expanded_col,
                        expanded_row,
                    )
                    continue
                context = _rotate_background_context(
                    expanded_context,
                    rng.uniform(0.0, 360.0),
                    image_size,
                )

            tile = context.tile
            nodata_mask = context.nodata_mask
            water_mask = context.water_mask
            surface_water_mask = context.surface_water_mask
            surface_valid_mask = context.surface_valid_mask
            berth_water_mask = context.berth_water_mask
            berth_segments = context.berth_segments
            berth_runs = context.berth_runs

            if is_dark_tile(tile):
                logger.debug(
                    "Dark tile in %s at col=%d row=%d (mean=%.1f) — retrying",
                    tif_path.name,
                    col,
                    row,
                    float(tile.mean()),
                )
                continue
            if nodata_mask.any():
                logger.debug(
                    "Tile in %s at col=%d row=%d contains blacked-out pixels — retrying",
                    tif_path.name,
                    col,
                    row,
                )
                continue

            water_mask = erode_mask(water_mask, erode_coast)

            water_ratio = water_mask.sum() / water_mask.size
            if surface_valid_mask.any():
                surface_ratio = float(surface_water_mask[surface_valid_mask].mean())
            else:
                surface_ratio = 0.0
            background_surface = classify_background_surface(surface_ratio)
            if water_ratio < min_water_ratio:
                continue
            if required_surface is not None and background_surface != required_surface:
                continue
            if water_ratio >= min_water_ratio:
                break
        else:
            try:
                if required_surface is not None:
                    return None
                if include_surface_category:
                    return tile, [], 0, background_surface  # type: ignore[possibly-undefined]
                return tile, [], 0  # type: ignore[possibly-undefined]
            except NameError:
                return None

    if debug_bg_color is None:
        tile = augment_tile(tile, rng)
    else:
        tile = np.zeros_like(tile)
        tile[:, :] = debug_bg_color

    blur_sigma = _edge_hardness_to_blur_sigma(edge_hardness)
    n_events = rng.randint(*ships_per_image)
    occupancy = np.zeros((image_size, image_size), dtype=bool)
    labels: list[str] = []
    n_clusters = 0
    single_ships: list[SingleShipPlacement] = []
    failed_placements = 0
    checked_failure_streak = False
    min_box_w, min_box_h = _minimum_ship_bbox(ship_resolution, ship_length_range)
    tile_shadow_azimuth = shadow_azimuth_rad if shadow_azimuth_rad is not None else rng.uniform(0.0, 2.0 * math.pi)
    shadow_length_min, shadow_length_max = shadow_length_range
    shadow_length_min, shadow_length_max = sorted((shadow_length_min, shadow_length_max))
    tile_shadow_length = rng.uniform(shadow_length_min, shadow_length_max)
    tile_shadow_alpha = _sample_shadow_alpha(rng) if shadow_alpha_scale > 0.0 else 0.0

    # Sample off-nadir geometry once per tile so all ships share the same
    # sensor viewing angle and world azimuth.
    tile_offnadir_deg = rng.uniform(*offnadir_range)
    tile_sensor_az_world_deg = rng.uniform(0.0, 360.0)

    for _ in range(n_events):
        is_cluster = rng.random() < cluster_prob
        placed_event = False

        if is_cluster:
            new_labels = _place_cluster(
                water_mask,
                occupancy,
                svg_metas,
                ship_resolution,
                rng,
                cluster_size,
                blur_sigma,
                ship_alpha,
                class_id,
                image_size,
                tile,
                length_range=ship_length_range,
                lb_ratio_range=ship_lb_ratio_range,
                length_exponent=length_exponent,
                size_thresholds=size_thresholds,
                mixed_prob=cluster_mixed_prob,
                shadow_azimuth_rad=tile_shadow_azimuth,
                shadow_length=tile_shadow_length,
                shadow_alpha=tile_shadow_alpha,
                shadow_alpha_scale=shadow_alpha_scale,
                berth_prob=berth_prob,
                berth_stern_prob=berth_stern_prob,
                berth_water_mask=berth_water_mask,
                berth_segments=berth_segments,
                berth_runs=berth_runs,
                offnadir_deg=tile_offnadir_deg,
                sensor_az_world_deg=tile_sensor_az_world_deg,
                shipgen_kwargs=shipgen_kwargs,
            )
            labels.extend(new_labels)
            if new_labels:
                n_clusters += 1
                placed_event = True
        else:
            if berth_water_mask is not None and berth_segments and rng.random() < max(0.0, min(1.0, berth_prob)):
                berthed_labels = _place_berthed_cluster(
                    berth_water_mask,
                    occupancy,
                    berth_segments,
                    svg_metas,
                    ship_resolution,
                    rng,
                    1,
                    ship_alpha,
                    class_id,
                    image_size,
                    tile,
                    ship_length_range,
                    length_exponent,
                    size_thresholds,
                    False,
                    berth_stern=rng.random() < max(0.0, min(1.0, berth_stern_prob)),
                    blur_sigma=blur_sigma,
                    shadow_azimuth_rad=tile_shadow_azimuth,
                    shadow_length=tile_shadow_length,
                    shadow_alpha=tile_shadow_alpha,
                    shadow_alpha_scale=shadow_alpha_scale,
                    offnadir_deg=tile_offnadir_deg,
                    sensor_az_world_deg=tile_sensor_az_world_deg,
                    shipgen_kwargs=shipgen_kwargs,
                    lb_ratio_range=ship_lb_ratio_range,
                    berth_runs=berth_runs,
                )
                if berthed_labels:
                    labels.extend(berthed_labels)
                    placed_event = True

            if not placed_event:
                angle_deg = rng.uniform(0, 360)
                sensor_az_ship_deg = (tile_sensor_az_world_deg - angle_deg) % 360.0
                svg_text = _pick_svg(
                    svg_metas, rng, ship_length_range, ship_lb_ratio_range,
                    tile_offnadir_deg,
                    sensor_az_ship_deg,
                    shipgen_kwargs=shipgen_kwargs,
                )
                angle_rad = math.radians(angle_deg)
                _cls_name, bw, lh, _lb = _resolve_ship_dimensions(
                    svg_text,
                    ship_resolution,
                    rng,
                    ship_length_range,
                    ship_lb_ratio_range,
                    length_exponent=length_exponent,
                )

                available = water_mask & ~occupancy
                pos = find_water_position(available, bw, lh, angle_rad, rng)
                if pos is not None:
                    rotated = _render_ship_from_dimensions(
                        svg_text,
                        bw,
                        lh,
                        blur_sigma,
                        angle_deg=angle_deg,
                        supersample=4,
                    )
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
                    cid = _ship_class_id(lh, ship_resolution, class_id, size_thresholds)
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
                    placed_event = True

        if placed_event:
            failed_placements = 0
            checked_failure_streak = False
            continue

        failed_placements += 1
        if failed_placements < _PLACEMENT_FAILURE_CHECK_INTERVAL or checked_failure_streak:
            continue

        available = water_mask & ~occupancy
        checked_failure_streak = True
        if not _has_full_open_box(available, min_box_w, min_box_h) and not _has_full_open_box(
            available,
            min_box_h,
            min_box_w,
        ):
            break

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
            occlusion_mask=_wake_occlusion_mask(occupancy, ship),
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
        blend_ship(
            tile,
            ship.rotated,
            ship.cx,
            ship.cy,
            ship.alpha,
            ship.water_tint,
        )
        labels.append(format_obb_label(ship.class_id, ship.corners, image_size, image_size))

    if include_surface_category:
        return tile, labels, n_clusters, background_surface
    return tile, labels, n_clusters