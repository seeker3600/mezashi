"""Spatial placement and cluster layout helpers for synthetic datagen."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from medetect.datagen.obb import compute_obb_corners, format_obb_label
from medetect.datagen.render import (
    extract_hull_fill,
    extract_hull_polygon,
    gaussian_blur_rgba_premultiplied,
    rasterize_ship_svg,
)
from medetect.datagen.scene import (
    RgbaLayerPatch,
    _blend_rgba_layer,
    _blend_rgba_patch,
    _cluster_scene_origin,
    _composite_rgba,
    _darken_rgba_layer,
    _darken_rgba_patch,
    _downsample_cluster_patch,
    _make_shadow_rgba,
    _rasterize_ship_scene,
    _sample_water_tint,
    _scene_patch_bounds,
    _shadow_alpha_for_ship,
    _shadow_blur_sigma,
    _shadow_offset_pixels,
)
from medetect.datagen.ship import (
    MIN_SHIP_BEAM_PX,
    MIN_SHIP_LENGTH_PX,
    SHIP_LENGTHS_M,
    _DEFAULT_LENGTH_M,
    _SvgMeta,
    _max_reasonable_lb_ratio,
    _pick_svg,
    _resolve_ship_dimensions,
    _scale_ship_pixel_size,
    _ship_class_id,
)
from medetect.datagen.svg import parse_svg_metadata

_CLUSTER_SCENE_SUPERSAMPLE: int = 4
_CLUSTER_RESAMPLE_PAD_OUTPUT_PX: int = 3

_BerthSegment = tuple[tuple[float, float], tuple[float, float]]
_BERTH_RUN_CONNECT_TOL_PX = 2.5
_BERTH_RUN_MAX_TURN_COS = math.cos(math.radians(20.0))
_BERTH_LAND_CLEARANCE_PX = 0.25


def _rgba_scene_rect(
    layer: NDArray[np.uint8],
    x0: int,
    y0: int,
) -> tuple[int, int, int, int]:
    """Return an RGBA layer rectangle in scene coordinates."""
    return x0, y0, x0 + layer.shape[1], y0 + layer.shape[0]


def _composite_items_to_patch(
    items: list[tuple[NDArray[np.uint8], int, int]],
    scene_size: int,
    scene_scale: int,
    *,
    padding: int = 0,
) -> RgbaLayerPatch | None:
    """Composite RGBA items into a local supersampled patch and downsample it."""
    if not items:
        return None

    bounds = [_rgba_scene_rect(layer, x0, y0) for layer, x0, y0 in items]
    patch_bounds = _scene_patch_bounds(bounds, scene_size, scene_scale, padding=padding)
    if patch_bounds is None:
        return None

    patch_x0, patch_y0, patch_x1, patch_y1 = patch_bounds
    patch = np.zeros((patch_y1 - patch_y0, patch_x1 - patch_x0, 4), dtype=np.uint8)
    for layer, x0, y0 in items:
        _composite_rgba(patch, layer, x0 - patch_x0, y0 - patch_y0)
    return _downsample_cluster_patch(patch, patch_x0, patch_y0, scene_scale)


@dataclass(frozen=True)
class _RaftShipPlacement:
    """Resolved raft-cluster ship placement used for final rendering."""

    svg_text: str
    cx: float
    cy: float
    bw: int
    lh: int
    angle_deg: float
    angle_rad: float
    class_id: int
    hull_geom: BaseGeometry
    hull_fill: tuple[int, int, int, int]


_BerthFrame = tuple[float, float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class _BerthRunSegment:
    segment: _BerthSegment
    frame: _BerthFrame
    start_s: float
    end_s: float


@dataclass(frozen=True)
class _BerthRun:
    segments: tuple[_BerthRunSegment, ...]
    length: float


def find_water_position(
    water_mask: NDArray[np.bool_],
    ship_w: int,
    ship_h: int,
    angle_rad: float,
    rng: random.Random,
    *,
    max_attempts: int = 100,
) -> tuple[int, int] | None:
    """Find a valid centre position for a ship on the water mask."""
    img_h, img_w = water_mask.shape

    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    bbox_w = int(ship_w * cos_a + ship_h * sin_a) + 2
    bbox_h = int(ship_w * sin_a + ship_h * cos_a) + 2
    half_w = bbox_w // 2
    half_h = bbox_h // 2

    if 2 * half_w >= img_w or 2 * half_h >= img_h:
        return None

    for _ in range(max_attempts):
        cx = rng.randint(half_w, img_w - half_w - 1)
        cy = rng.randint(half_h, img_h - half_h - 1)
        region = water_mask[cy - half_h : cy + half_h, cx - half_w : cx + half_w]
        if region.all():
            return cx, cy

    return None


def _iter_polygon_geometries(geometry: BaseGeometry):
    """Yield polygonal components from a Shapely geometry."""
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            yield poly
        return
    if isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from _iter_polygon_geometries(child)


@lru_cache(maxsize=4096)
def _normalized_local_hull_geometry(
    svg_text: str,
) -> BaseGeometry:
    """Return the ship hull geometry normalized to a unit beam/length box."""
    _ship_class, lb_ratio = parse_svg_metadata(svg_text)
    hull_points = extract_hull_polygon(svg_text)
    pts = [
        (x - 0.5, y / max(lb_ratio, 1e-6) - 0.5)
        for x, y in hull_points
    ]
    return Polygon(pts).buffer(0)


@lru_cache(maxsize=4096)
def _base_local_hull_geometry(
    svg_text: str,
    beam_px: int,
    length_px: int,
) -> BaseGeometry:
    """Return the unrotated ship hull geometry in pixel units centred at the origin."""
    return affinity.scale(
        _normalized_local_hull_geometry(svg_text),
        xfact=float(beam_px),
        yfact=float(length_px),
        origin=(0.0, 0.0),
    )


@lru_cache(maxsize=4096)
def _local_hull_geometry(
    svg_text: str,
    beam_px: int,
    length_px: int,
    angle_deg: float,
) -> BaseGeometry:
    """Return the ship hull geometry in pixel units centred at the origin."""
    geometry = _base_local_hull_geometry(svg_text, beam_px, length_px)
    if angle_deg != 0.0:
        geometry = affinity.rotate(geometry, angle_deg, origin=(0.0, 0.0), use_radians=False)
    return geometry


def _translate_hull_geometry(
    geometry: BaseGeometry,
    cx: float,
    cy: float,
) -> BaseGeometry:
    """Translate a local hull geometry to image-space centre coordinates."""
    return affinity.translate(geometry, xoff=cx, yoff=cy)


def _geometry_projection_extents(
    geometry: BaseGeometry,
    axis_x: float,
    axis_y: float,
) -> tuple[float, float]:
    """Project a polygon geometry onto a row axis and return min/max extents."""
    values: list[float] = []
    for poly in _iter_polygon_geometries(geometry):
        coords = np.asarray(poly.exterior.coords[:-1], dtype=np.float32)
        if coords.size == 0:
            continue
        values.extend((coords[:, 0] * axis_x + coords[:, 1] * axis_y).tolist())
    if not values:
        return 0.0, 0.0
    return float(min(values)), float(max(values))


def _signed_geometry_gap(
    geometry_a: BaseGeometry,
    geometry_b: BaseGeometry,
) -> float:
    """Signed hull gap: negative for overlap, zero for touch, positive for separation."""
    intersection = geometry_a.intersection(geometry_b)
    inter_area = float(getattr(intersection, "area", 0.0))
    if inter_area > 1e-9:
        return -math.sqrt(inter_area)
    distance = float(geometry_a.distance(geometry_b))
    if distance <= 1e-9:
        return 0.0
    return distance


def _draw_geometry_fill(
    image: Image.Image,
    geometry: BaseGeometry,
    fill: tuple[int, int, int, int],
    *,
    scale: float = 1.0,
) -> None:
    """Rasterize a polygonal geometry as a filled layer onto an RGBA image."""
    draw = ImageDraw.Draw(image)
    for poly in _iter_polygon_geometries(geometry):
        exterior = [(x * scale, y * scale) for x, y in poly.exterior.coords]
        draw.polygon(exterior, fill=fill)


def _shadow_source_rgba(
    detail_rgba: NDArray[np.uint8],
    scaled_hull: BaseGeometry,
    x0_scene: int,
    y0_scene: int,
) -> NDArray[np.uint8]:
    """Build a shadow silhouette from detail alpha plus the hull polygon."""
    shadow_source = np.zeros_like(detail_rgba)
    shadow_source[:, :, 3] = detail_rgba[:, :, 3]

    hull_alpha = Image.new("L", (detail_rgba.shape[1], detail_rgba.shape[0]), 0)
    translated_hull = affinity.translate(scaled_hull, xoff=-x0_scene, yoff=-y0_scene)
    draw = ImageDraw.Draw(hull_alpha)
    for poly in _iter_polygon_geometries(translated_hull):
        draw.polygon(list(poly.exterior.coords), fill=255)

    np.maximum(
        shadow_source[:, :, 3],
        np.array(hull_alpha, dtype=np.uint8),
        out=shadow_source[:, :, 3],
    )
    return shadow_source


def _tight_cluster_bridge_geometry(
    ships: list[_RaftShipPlacement],
    join_tolerance: float,
) -> BaseGeometry | None:
    """Return only the interior geometry needed to close tight-cluster slits.

    The previous tight-cluster underlay buffered the whole merged hull group,
    which also painted a coloured ring around the outer cluster boundary.
    Here we perform a morphological closing on the unioned hulls and keep only
    the newly added bridge area, so the helper fills internal seams without
    expanding the exterior silhouette.
    """
    if join_tolerance <= 0.0 or len(ships) < 2:
        return None

    merged_hull: BaseGeometry | None = None
    for ship in ships:
        merged_hull = ship.hull_geom if merged_hull is None else merged_hull.union(ship.hull_geom)

    if merged_hull is None or merged_hull.is_empty:
        return None

    closed_hull = merged_hull.buffer(join_tolerance).buffer(-join_tolerance)
    if closed_hull.is_empty:
        return None

    bridge = closed_hull.difference(merged_hull).buffer(0)
    if bridge.is_empty or float(getattr(bridge, "area", 0.0)) <= 1e-6:
        return None
    return bridge


def _stamp_geometry_occupancy(
    occupancy: NDArray[np.bool_],
    geometry: BaseGeometry,
    margin: float = 1.5,
) -> None:
    """Mark a polygonal hull footprint as occupied on the raster occupancy map."""
    stamped = geometry.buffer(margin) if margin > 0 else geometry
    img = Image.fromarray(occupancy.astype(np.uint8) * 255)
    for poly in _iter_polygon_geometries(stamped):
        ImageDraw.Draw(img).polygon(list(poly.exterior.coords), fill=255)
    occupancy[:] = np.array(img) > 0


def _geometry_hits_mask(
    mask: NDArray[np.bool_],
    geometry: BaseGeometry,
) -> bool:
    """Return True when a polygonal geometry overlaps any True pixels in mask."""
    if geometry.is_empty:
        return False

    min_x, min_y, max_x, max_y = geometry.bounds
    x0 = max(0, math.floor(min_x) - 1)
    y0 = max(0, math.floor(min_y) - 1)
    x1 = min(mask.shape[1], math.ceil(max_x) + 2)
    y1 = min(mask.shape[0], math.ceil(max_y) + 2)
    if x0 >= x1 or y0 >= y1:
        return False

    patch_mask = mask[y0:y1, x0:x1]
    if not patch_mask.any():
        return False

    img = Image.new("L", (x1 - x0, y1 - y0), 0)
    draw = ImageDraw.Draw(img)
    for poly in _iter_polygon_geometries(geometry):
        exterior = [(x - x0, y - y0) for x, y in poly.exterior.coords]
        draw.polygon(exterior, fill=255)
    geom_mask = np.array(img) > 0
    return bool((geom_mask & patch_mask).any())


def _geometry_intrudes_land(
    water_mask: NDArray[np.bool_],
    hull_geom: BaseGeometry,
    clearance_px: float = _BERTH_LAND_CLEARANCE_PX,
) -> bool:
    """Return True when the hull interior penetrates land beyond a contact tolerance."""
    inner_geom = hull_geom.buffer(-clearance_px, join_style=2) if clearance_px > 0 else hull_geom
    if inner_geom.is_empty:
        inner_geom = hull_geom.representative_point().buffer(0.05)
    land_mask = ~water_mask
    return _geometry_hits_mask(land_mask, inner_geom)


_ORIGINAL_GEOMETRY_INTRUDES_LAND = _geometry_intrudes_land


def _resolve_berth_land_intrusion(
    water_mask: NDArray[np.bool_],
    hull_geom: BaseGeometry,
    cx: float,
    cy: float,
    water_nx: float,
    water_ny: float,
    *,
    max_shift_px: float,
    step_px: float = 0.25,
) -> tuple[float, float, BaseGeometry] | None:
    """Push a berth candidate offshore until it clears land intrusion.

    This keeps dock contact whenever possible while preventing visible land
    penetration on curved or pixel-stepped shorelines.
    """
    max_shift_px = max(0.0, max_shift_px)
    precision_px = max(step_px, 1e-6)
    use_fast_intrusion_check = _geometry_intrudes_land is _ORIGINAL_GEOMETRY_INTRUDES_LAND
    land_mask = ~water_mask
    inner_hull_geom = hull_geom.buffer(-_BERTH_LAND_CLEARANCE_PX, join_style=2)
    if inner_hull_geom.is_empty:
        inner_hull_geom = hull_geom.representative_point().buffer(0.05)

    evaluated: dict[float, tuple[float, float, BaseGeometry, bool]] = {}

    def _evaluate_shift(shift: float) -> tuple[float, float, BaseGeometry, bool]:
        clamped_shift = min(max_shift_px, max(0.0, shift))
        cached = evaluated.get(clamped_shift)
        if cached is not None:
            return cached

        if clamped_shift <= 0.0:
            shifted_geom = hull_geom
            shifted_inner_geom = inner_hull_geom
        else:
            xoff = water_nx * clamped_shift
            yoff = water_ny * clamped_shift
            shifted_geom = affinity.translate(
                hull_geom,
                xoff=xoff,
                yoff=yoff,
            )
            shifted_inner_geom = affinity.translate(
                inner_hull_geom,
                xoff=xoff,
                yoff=yoff,
            )
        result = (
            cx + water_nx * clamped_shift,
            cy + water_ny * clamped_shift,
            shifted_geom,
            _geometry_hits_mask(land_mask, shifted_inner_geom)
            if use_fast_intrusion_check
            else _geometry_intrudes_land(water_mask, shifted_geom),
        )
        evaluated[clamped_shift] = result
        return result

    base_cx, base_cy, base_geom, intrudes = _evaluate_shift(0.0)
    if not intrudes:
        return base_cx, base_cy, base_geom
    if max_shift_px <= 0.0:
        return None

    coarse_step = min(
        max_shift_px,
        max(precision_px * 4.0, max_shift_px / 4.0),
    )
    coarse_step = max(coarse_step, precision_px)

    intrusive_shift = 0.0
    clear_shift: float | None = None
    probe_shift = coarse_step
    while probe_shift < max_shift_px:
        _probe_cx, _probe_cy, _probe_geom, probe_intrudes = _evaluate_shift(probe_shift)
        if not probe_intrudes:
            clear_shift = probe_shift
            break
        intrusive_shift = probe_shift
        probe_shift += coarse_step

    if clear_shift is None:
        end_cx, end_cy, end_geom, end_intrudes = _evaluate_shift(max_shift_px)
        if end_intrudes:
            return None
        clear_shift = max_shift_px
        clear_result = (end_cx, end_cy, end_geom)
    else:
        clear_cx, clear_cy, clear_geom, _clear_intrudes = _evaluate_shift(clear_shift)
        clear_result = (clear_cx, clear_cy, clear_geom)

    while clear_shift - intrusive_shift > precision_px:
        probe_shift = (intrusive_shift + clear_shift) / 2.0
        probe_cx, probe_cy, probe_geom, probe_intrudes = _evaluate_shift(probe_shift)
        if probe_intrudes:
            intrusive_shift = probe_shift
            continue
        clear_shift = probe_shift
        clear_result = (probe_cx, probe_cy, probe_geom)

    return clear_result


def _render_vector_raft_cluster(
    ships: list[_RaftShipPlacement],
    image_size: int,
    blur_sigma: float,
    scene_scale: int,
    join_tolerance: float = 0.0,
    shadow_azimuth_rad: float | None = None,
    shadow_length: float | None = None,
    shadow_alpha: float = 0.0,
    shadow_alpha_scale: float = 1.0,
) -> tuple[RgbaLayerPatch | None, RgbaLayerPatch]:
    """Render a raft cluster from vector hulls plus per-ship detail layers."""
    scene_size = image_size * scene_scale
    resample_pad = scene_scale * _CLUSTER_RESAMPLE_PAD_OUTPUT_PX
    blur_pad = 0
    if blur_sigma > 0 and ships:
        blur_pad = math.ceil(blur_sigma * scene_scale * 3.0) + 2

    hull_entries: list[tuple[BaseGeometry, tuple[int, int, int, int]]] = []
    layer_bounds: list[tuple[float, float, float, float]] = []

    if join_tolerance > 0.0 and ships:
        bridge_geom = _tight_cluster_bridge_geometry(ships, join_tolerance)
        if bridge_geom is not None and not bridge_geom.is_empty:
            avg_fill = tuple(
                round(sum(ship.hull_fill[idx] for ship in ships) / len(ships))
                for idx in range(4)
            )
            scaled_underlay = affinity.scale(
                bridge_geom,
                xfact=scene_scale,
                yfact=scene_scale,
                origin=(0.0, 0.0),
            )
            hull_entries.append((scaled_underlay, avg_fill))
            layer_bounds.append(tuple(map(float, scaled_underlay.bounds)))
    for ship in ships:
        scaled_hull = affinity.scale(
            ship.hull_geom,
            xfact=scene_scale,
            yfact=scene_scale,
            origin=(0.0, 0.0),
        )
        hull_entries.append((scaled_hull, ship.hull_fill))
        layer_bounds.append(tuple(map(float, scaled_hull.bounds)))

    detail_items: list[tuple[NDArray[np.uint8], int, int]] = []
    shadow_items: list[tuple[NDArray[np.uint8], int, int]] = []
    render_shadows = (
        shadow_alpha > 0.0
        and shadow_alpha_scale > 0.0
        and shadow_azimuth_rad is not None
        and shadow_length is not None
    )

    for ship, (scaled_hull, _fill) in zip(ships, hull_entries[-len(ships):], strict=True):
        detail_rgba = rasterize_ship_svg(
            ship.svg_text,
            max(1, ship.bw * scene_scale),
            max(1, ship.lh * scene_scale),
            angle_deg=ship.angle_deg,
            supersample=1,
            exclude_hull=True,
        )
        x0_scene, y0_scene = _cluster_scene_origin(
            ship.cx,
            ship.cy,
            detail_rgba,
            scene_scale,
        )
        detail_items.append((detail_rgba, x0_scene, y0_scene))
        layer_bounds.append(_rgba_scene_rect(detail_rgba, x0_scene, y0_scene))

        if render_shadows:
            shadow_source = _shadow_source_rgba(detail_rgba, scaled_hull, x0_scene, y0_scene)
            offset_x, offset_y = _shadow_offset_pixels(
                ship.bw,
                ship.lh,
                shadow_azimuth_rad,
                shadow_length,
                scene_scale=scene_scale,
            )
            cast_length = math.hypot(offset_x, offset_y)
            shadow_rgba = _make_shadow_rgba(
                shadow_source,
                offset_x=offset_x,
                offset_y=offset_y,
                blur_sigma=_shadow_blur_sigma(
                    ship.bw,
                    ship.lh,
                    cast_length,
                    scene_scale=scene_scale,
                ),
                alpha_scale=_shadow_alpha_for_ship(ship.bw, ship.lh),
            )
            shadow_x0, shadow_y0 = _cluster_scene_origin(
                ship.cx,
                ship.cy,
                shadow_rgba,
                scene_scale,
            )
            shadow_items.append((shadow_rgba, shadow_x0, shadow_y0))

    patch_bounds = _scene_patch_bounds(
        layer_bounds,
        scene_size,
        scene_scale,
        padding=blur_pad + resample_pad,
    )
    if patch_bounds is None:
        return None, RgbaLayerPatch(0, 0, np.zeros((1, 1, 4), dtype=np.uint8))

    patch_x0, patch_y0, patch_x1, patch_y1 = patch_bounds
    hull_img = Image.new("RGBA", (patch_x1 - patch_x0, patch_y1 - patch_y0), (0, 0, 0, 0))
    for geometry, fill in hull_entries:
        translated = affinity.translate(geometry, xoff=-patch_x0, yoff=-patch_y0)
        _draw_geometry_fill(hull_img, translated, fill)

    layer = np.array(hull_img, dtype=np.uint8)
    for detail_rgba, x0_scene, y0_scene in detail_items:
        _composite_rgba(layer, detail_rgba, x0_scene - patch_x0, y0_scene - patch_y0)

    if blur_sigma > 0 and ships:
        layer = gaussian_blur_rgba_premultiplied(
            layer,
            blur_sigma * scene_scale,
        )

    return (
        _composite_items_to_patch(
            shadow_items,
            scene_size,
            scene_scale,
            padding=resample_pad,
        ),
        _downsample_cluster_patch(layer, patch_x0, patch_y0, scene_scale),
    )


def _obb_aabb_bounds(
    cx: float,
    cy: float,
    w: int,
    h: int,
    angle_rad: float,
    image_size: int,
    *,
    padding: float = 1.0,
) -> tuple[int, int, int, int]:
    """Return clipped axis-aligned bounds for an OBB centred at ``(cx, cy)``."""
    cos_abs = abs(math.cos(angle_rad))
    sin_abs = abs(math.sin(angle_rad))
    half_x = (w * cos_abs + h * sin_abs) / 2.0 + padding
    half_y = (w * sin_abs + h * cos_abs) / 2.0 + padding
    x0 = max(0, math.floor(cx - half_x))
    x1 = min(image_size, math.ceil(cx + half_x))
    y0 = max(0, math.floor(cy - half_y))
    y1 = min(image_size, math.ceil(cy + half_y))
    return x0, y0, x1, y1


def _stamp_occupancy(
    occupancy: NDArray[np.bool_],
    cx: float,
    cy: float,
    w: int,
    h: int,
    angle_rad: float,
    margin: int = 2,
) -> None:
    """Mark a rotated OBB footprint (+margin px) as occupied in-place."""
    corners = compute_obb_corners(
        float(cx),
        float(cy),
        float(w + margin * 2),
        float(h + margin * 2),
        angle_rad,
    )
    img = Image.fromarray(occupancy.astype(np.uint8) * 255)
    draw = ImageDraw.Draw(img)
    draw.polygon(corners, fill=255)
    occupancy[:] = np.array(img) > 0


def _obb_on_water(
    water_mask: NDArray[np.bool_],
    cx: float,
    cy: float,
    w: int,
    h: int,
    angle_rad: float,
    required: int = 4,
) -> bool:
    """Return True if the OBB is sufficiently on water."""
    corners = compute_obb_corners(float(cx), float(cy), float(w), float(h), angle_rad)
    h_mask, w_mask = water_mask.shape
    points = list(corners) + [(float(cx), float(cy))]
    on_water = 0
    for x, y in points:
        xi, yi = round(x), round(y)
        if 0 <= xi < w_mask and 0 <= yi < h_mask and water_mask[yi, xi]:
            on_water += 1
    return on_water >= required


def _obb_on_berth_water(
    water_mask: NDArray[np.bool_],
    cx: float,
    cy: float,
    w: int,
    h: int,
    angle_rad: float,
    water_nx: float,
    water_ny: float,
) -> bool:
    """Return True if a berthed OBB keeps its offshore side over water.

    Unlike open-water placement, dock-side corners are allowed to graze land so
    long as the ship centre and at least one offshore corner stay on water.
    """
    if not _mask_contains(water_mask, cx, cy):
        return False

    corners = compute_obb_corners(float(cx), float(cy), float(w), float(h), angle_rad)
    corner_samples: list[tuple[float, bool]] = []
    on_water = 1
    for x, y in corners:
        wet = _mask_contains(water_mask, x, y)
        if wet:
            on_water += 1
        corner_samples.append((((x - cx) * water_nx) + ((y - cy) * water_ny), wet))

    offshore = sorted(corner_samples, key=lambda item: item[0], reverse=True)[:2]
    if len(offshore) < 2 or not any(wet for _score, wet in offshore):
        return False
    return on_water >= 2


def _mask_contains(
    mask: NDArray[np.bool_],
    x: float,
    y: float,
) -> bool:
    xi, yi = round(x), round(y)
    return 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi])


def _angle_deg_from_stern_direction(
    stern_dx: float,
    stern_dy: float,
) -> float:
    return math.degrees(math.atan2(-stern_dx, stern_dy))


def _minimum_berthed_ship_span_px(
    resolution_m: float,
    length_range: tuple[float, float] | None,
    berth_stern: bool,
) -> float:
    """Return a safe lower bound on berth tangent span for one ship."""
    if length_range is not None:
        min_length_m = max(length_range[0], 1.0)
    else:
        class_mins = [lengths[0] for lengths in SHIP_LENGTHS_M.values()]
        min_length_m = min(class_mins + [_DEFAULT_LENGTH_M[0]])

    min_length_px = max(MIN_SHIP_LENGTH_PX, round(min_length_m / max(resolution_m, 1e-6)))
    if not berth_stern:
        return float(min_length_px)

    max_lb_ratio = _max_reasonable_lb_ratio(min_length_m)
    min_beam_m = min_length_m / max(max_lb_ratio, 1e-6)
    min_beam_px = max(MIN_SHIP_BEAM_PX, round(min_beam_m / max(resolution_m, 1e-6)))
    return float(min_beam_px)


def _max_berthed_ships_for_run(
    run_length: float,
    ship_gap: float,
    resolution_m: float,
    length_range: tuple[float, float] | None,
    berth_stern: bool,
) -> int:
    """Return the maximum ship count that can fit by physical lower bounds."""
    min_ship_span = _minimum_berthed_ship_span_px(
        resolution_m,
        length_range,
        berth_stern,
    )
    if run_length + ship_gap < min_ship_span:
        return 0
    return max(0, math.floor((run_length + ship_gap) / (min_ship_span + ship_gap)))


def _segment_endpoint_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _berth_segment_frame(
    segment: _BerthSegment,
    water_mask: NDArray[np.bool_],
) -> _BerthFrame | None:
    (x0, y0), (x1, y1) = segment
    dx = x1 - x0
    dy = y1 - y0
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return None

    tx = dx / seg_len
    ty = dy / seg_len
    mid_x = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0
    sample_dist = 2.0
    candidates = [(-ty, tx), (ty, -tx)]

    best: tuple[float, float] | None = None
    best_score = -1
    for nx, ny in candidates:
        water_score = 0
        land_score = 0
        for offset in (sample_dist, sample_dist * 2.0, sample_dist * 3.0):
            if _mask_contains(water_mask, mid_x + nx * offset, mid_y + ny * offset):
                water_score += 1
            if _mask_contains(water_mask, mid_x - nx * offset, mid_y - ny * offset):
                land_score += 1
        score = water_score - land_score
        if water_score > 0 and score > best_score:
            best = (nx, ny)
            best_score = score

    if best is None or best_score <= 0:
        return None

    nx, ny = best
    return x0, y0, tx, ty, nx, ny, seg_len, mid_x, mid_y


def _orient_berth_segment_for_connection(
    segment: _BerthSegment,
    endpoint: tuple[float, float],
) -> _BerthSegment | None:
    if _segment_endpoint_distance(segment[0], endpoint) <= _BERTH_RUN_CONNECT_TOL_PX:
        return segment
    if _segment_endpoint_distance(segment[1], endpoint) <= _BERTH_RUN_CONNECT_TOL_PX:
        return (segment[1], segment[0])
    return None


def _merge_berth_segments(
    berth_segments: list[_BerthSegment],
    water_mask: NDArray[np.bool_],
) -> list[_BerthSegment]:
    """Return connected berth segments with stable orientation preserved."""
    merged: list[_BerthSegment] = []
    current_run: list[tuple[_BerthSegment, _BerthFrame]] = []

    for raw_segment in berth_segments:
        segment = raw_segment
        if current_run:
            oriented = _orient_berth_segment_for_connection(raw_segment, current_run[-1][0][1])
            if oriented is not None:
                segment = oriented

        frame = _berth_segment_frame(segment, water_mask)
        if frame is None:
            if current_run:
                merged.extend(run_segment for run_segment, _run_frame in current_run)
                current_run = []
            continue

        if not current_run:
            current_run = [(segment, frame)]
            continue

        prev_segment, prev_frame = current_run[-1]
        connected = _segment_endpoint_distance(segment[0], prev_segment[1]) <= _BERTH_RUN_CONNECT_TOL_PX
        tangent_ok = (prev_frame[2] * frame[2]) + (prev_frame[3] * frame[3]) >= _BERTH_RUN_MAX_TURN_COS
        normal_ok = (prev_frame[4] * frame[4]) + (prev_frame[5] * frame[5]) > 0.0
        if connected and tangent_ok and normal_ok:
            current_run.append((segment, frame))
            continue

        merged.extend(run_segment for run_segment, _run_frame in current_run)
        current_run = [(segment, frame)]

    if current_run:
        merged.extend(run_segment for run_segment, _run_frame in current_run)
    return merged if merged else list(berth_segments)


def _build_berth_runs(
    berth_segments: list[_BerthSegment],
    water_mask: NDArray[np.bool_],
) -> list[_BerthRun]:
    """Group connected shoreline segments into berth runs while preserving bends."""
    runs: list[_BerthRun] = []
    current_run: list[tuple[_BerthSegment, _BerthFrame]] = []

    def _flush_run() -> None:
        nonlocal current_run
        if not current_run:
            return
        start_s = 0.0
        run_segments: list[_BerthRunSegment] = []
        for run_segment, run_frame in current_run:
            seg_len = run_frame[6]
            run_segments.append(
                _BerthRunSegment(
                    segment=run_segment,
                    frame=run_frame,
                    start_s=start_s,
                    end_s=start_s + seg_len,
                )
            )
            start_s += seg_len
        runs.append(_BerthRun(tuple(run_segments), start_s))
        current_run = []

    for raw_segment in berth_segments:
        segment = raw_segment
        if current_run:
            oriented = _orient_berth_segment_for_connection(raw_segment, current_run[-1][0][1])
            if oriented is not None:
                segment = oriented

        frame = _berth_segment_frame(segment, water_mask)
        if frame is None:
            _flush_run()
            continue

        if not current_run:
            current_run = [(segment, frame)]
            continue

        prev_segment, prev_frame = current_run[-1]
        connected = _segment_endpoint_distance(segment[0], prev_segment[1]) <= _BERTH_RUN_CONNECT_TOL_PX
        tangent_ok = (prev_frame[2] * frame[2]) + (prev_frame[3] * frame[3]) >= _BERTH_RUN_MAX_TURN_COS
        normal_ok = (prev_frame[4] * frame[4]) + (prev_frame[5] * frame[5]) > 0.0
        if connected and tangent_ok and normal_ok:
            current_run.append((segment, frame))
            continue

        _flush_run()
        current_run = [(segment, frame)]

    _flush_run()
    return runs


def _sample_berth_run(
    run: _BerthRun,
    distance_s: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Sample point and local frame at arclength distance along a berth run."""
    if not run.segments:
        return None

    target_s = max(0.0, min(distance_s, run.length))
    last_index = len(run.segments) - 1
    for index, run_segment in enumerate(run.segments):
        if target_s <= run_segment.end_s or index == last_index:
            x0, y0, tan_x, tan_y, water_nx, water_ny, seg_len, _mid_x, _mid_y = run_segment.frame
            local_s = max(0.0, min(seg_len, target_s - run_segment.start_s))
            return (
                x0 + tan_x * local_s,
                y0 + tan_y * local_s,
                tan_x,
                tan_y,
                water_nx,
                water_ny,
            )
    return None


def _offshore_contact_candidate(
    prev_hull: BaseGeometry,
    prev_cx: float,
    prev_cy: float,
    local_hull: BaseGeometry,
    offshore_nx: float,
    offshore_ny: float,
    tangent_x: float,
    tangent_y: float,
    bw: int,
    lh: int,
    angle_rad: float,
    image_size: int,
    water_mask: NDArray[np.bool_],
    pre_occupancy: NDArray[np.bool_],
    *,
    tangent_offset: float = 0.0,
) -> tuple[float, float, BaseGeometry] | None:
    """Return a raft-tight-style offshore contact candidate for alongside berth followers."""
    prev_min_offshore, prev_max_offshore = _geometry_projection_extents(
        prev_hull,
        offshore_nx,
        offshore_ny,
    )
    local_min_offshore, local_max_offshore = _geometry_projection_extents(
        local_hull,
        offshore_nx,
        offshore_ny,
    )
    prev_center_offshore = (prev_cx * offshore_nx) + (prev_cy * offshore_ny)
    prev_proj_half = (prev_max_offshore - prev_min_offshore) / 2.0
    proj_half = (local_max_offshore - local_min_offshore) / 2.0
    obb_penetration_limit = max(1.0, min(prev_proj_half, proj_half) * 0.22)
    base_contact_offset = prev_max_offshore - prev_center_offshore - local_min_offshore
    local_min_x, local_min_y, local_max_x, local_max_y = map(float, local_hull.bounds)
    scene_scale = _CLUSTER_SCENE_SUPERSAMPLE
    samples: list[tuple[float, float, float, float, BaseGeometry]] = []

    def _sample_contact(
        offshore_offset: float,
    ) -> tuple[float, float, float, BaseGeometry] | None:
        candidate_center_offshore = prev_center_offshore + offshore_offset
        penetration = prev_max_offshore - (candidate_center_offshore + local_min_offshore)
        obb_penetration = (prev_center_offshore + prev_proj_half) - (
            candidate_center_offshore - proj_half
        )
        if penetration > 1.0 or obb_penetration > obb_penetration_limit:
            return None

        cx = prev_cx + offshore_offset * offshore_nx + tangent_offset * tangent_x
        cy = prev_cy + offshore_offset * offshore_ny + tangent_offset * tangent_y
        if (
            local_min_x + cx < 0.0
            or local_min_y + cy < 0.0
            or local_max_x + cx > image_size
            or local_max_y + cy > image_size
        ):
            return None

        if not _obb_on_water(water_mask, cx, cy, bw, lh, angle_rad):
            return None

        cx0, cy0, cx1, cy1 = _obb_aabb_bounds(
            cx,
            cy,
            bw,
            lh,
            angle_rad,
            image_size,
            padding=0.0,
        )
        if pre_occupancy[cy0:cy1, cx0:cx1].any():
            return None

        hull_geom = _translate_hull_geometry(local_hull, cx, cy)
        signed_gap = _signed_geometry_gap(prev_hull, hull_geom)
        return signed_gap, cx, cy, hull_geom

    for adjust_steps in range(-2 * scene_scale, 7 * scene_scale + 1):
        offshore_offset = base_contact_offset + adjust_steps / scene_scale
        sampled = _sample_contact(offshore_offset)
        if sampled is None:
            continue
        signed_gap, cx, cy, hull_geom = sampled
        samples.append((offshore_offset, signed_gap, cx, cy, hull_geom))

    if not samples:
        return None

    samples.sort(key=lambda item: item[0])
    bracket: tuple[
        tuple[float, float, float, float, BaseGeometry],
        tuple[float, float, float, float, BaseGeometry],
    ] | None = None
    for left, right in zip(samples, samples[1:]):
        if left[1] <= 0.0 <= right[1]:
            bracket = (left, right)
            break
        if right[1] <= 0.0 <= left[1]:
            bracket = (right, left)
            break

    if bracket is not None:
        best = min(
            bracket,
            key=lambda item: (item[1] > 0.0, abs(item[1])),
        )
        low_offset = bracket[0][0]
        high_offset = bracket[1][0]
        for _ in range(12):
            mid_offset = (low_offset + high_offset) / 2.0
            sampled = _sample_contact(mid_offset)
            if sampled is None:
                break
            mid_gap, mid_cx, mid_cy, mid_geom = sampled
            candidate = (mid_offset, mid_gap, mid_cx, mid_cy, mid_geom)
            if (mid_gap <= 0.0 and abs(mid_gap) < abs(best[1])) or (
                best[1] > 0.0 and abs(mid_gap) < abs(best[1])
            ):
                best = candidate
            if mid_gap <= 0.0:
                low_offset = mid_offset
            else:
                high_offset = mid_offset
        return best[2], best[3], best[4]

    _offshore_offset, _signed_gap, cx, cy, hull_geom = min(
        samples,
        key=lambda item: (item[1] > 0.0, abs(item[1]), abs(item[0] - base_contact_offset)),
    )
    return cx, cy, hull_geom


def _place_alongside_berthed_run(
    run: _BerthRun,
    mid_sample: tuple[float, float, float, float, float, float],
    berth_water_mask: NDArray[np.bool_],
    occupancy: NDArray[np.bool_],
    svg_metas: list[_SvgMeta] | None,
    resolution_m: float,
    rng: random.Random,
    n_ships: int,
    class_id: int,
    image_size: int,
    length_range: tuple[float, float] | None,
    length_exponent: float,
    size_thresholds: tuple[float, ...] | None,
    mixed: bool,
    offnadir_deg: float,
    sensor_az_world_deg: float,
    shipgen_kwargs: dict[str, Any] | None = None,
) -> tuple[list[_RaftShipPlacement], NDArray[np.bool_]] | None:
    """Place a shoreline lead ship plus raft-tight offshore followers."""
    if n_ships <= 0 or run.length <= 1e-6:
        return None

    if run.length < _minimum_berthed_ship_span_px(
        resolution_m,
        length_range,
        berth_stern=False,
    ):
        return None

    mid_x, mid_y, mid_tan_x, mid_tan_y, _mid_water_nx, _mid_water_ny = mid_sample
    lead_angle_deg = _angle_deg_from_stern_direction(mid_tan_x, mid_tan_y)
    lead_sensor_az = (sensor_az_world_deg - lead_angle_deg) % 360.0

    if not mixed:
        svg_text_ref = _pick_svg(
            svg_metas,
            rng,
            length_range,
            offnadir_deg,
            lead_sensor_az,
            shipgen_kwargs=shipgen_kwargs,
        )
        _cls0, bw0, lh0, _lb0 = _resolve_ship_dimensions(
            svg_text_ref,
            resolution_m,
            rng,
            length_range,
            length_exponent,
        )

    if mixed:
        lead_svg_text = _pick_svg(
            svg_metas,
            rng,
            length_range,
            offnadir_deg,
            lead_sensor_az,
            shipgen_kwargs=shipgen_kwargs,
        )
        _lead_cls_name, lead_bw, lead_lh, _lead_lb = _resolve_ship_dimensions(
            lead_svg_text,
            resolution_m,
            rng,
            length_range,
            length_exponent,
        )
    else:
        lead_svg_text = svg_text_ref
        lead_bw, lead_lh = bw0, lh0

    lead_local_hull = _local_hull_geometry(lead_svg_text, lead_bw, lead_lh, lead_angle_deg)
    lead_min_t, lead_max_t = _geometry_projection_extents(lead_local_hull, mid_tan_x, mid_tan_y)
    lead_span = lead_max_t - lead_min_t
    if lead_span > run.length:
        return None

    lead_center_s = ((run.length - lead_span) / 2.0) - lead_min_t
    lead_sample = _sample_berth_run(run, lead_center_s)
    if lead_sample is None:
        return None

    lead_x, lead_y, tan_x, tan_y, water_nx, water_ny = lead_sample
    lead_angle_deg = _angle_deg_from_stern_direction(tan_x, tan_y)
    lead_angle_rad = math.radians(lead_angle_deg)
    lead_local_hull = _local_hull_geometry(lead_svg_text, lead_bw, lead_lh, lead_angle_deg)
    lead_min_n, _lead_max_n = _geometry_projection_extents(lead_local_hull, water_nx, water_ny)
    lead_water_offset = -lead_min_n
    lead_cx = lead_x + water_nx * lead_water_offset
    lead_cy = lead_y + water_ny * lead_water_offset
    lead_hull_geom = _translate_hull_geometry(lead_local_hull, lead_cx, lead_cy)
    lead_hull_fill = extract_hull_fill(lead_svg_text)

    resolved = _resolve_berth_land_intrusion(
        berth_water_mask,
        lead_hull_geom,
        lead_cx,
        lead_cy,
        water_nx,
        water_ny,
        max_shift_px=max(14.0, lead_water_offset + 2.0),
    )
    if resolved is None:
        return None

    lead_cx, lead_cy, lead_hull_geom = resolved
    min_x, min_y, max_x, max_y = lead_hull_geom.bounds
    if min_x < 0.0 or min_y < 0.0 or max_x > image_size or max_y > image_size:
        return None

    if not _obb_on_berth_water(
        berth_water_mask,
        lead_cx,
        lead_cy,
        lead_bw,
        lead_lh,
        lead_angle_rad,
        water_nx,
        water_ny,
    ):
        return None

    pre_occupancy = occupancy.copy()
    staged_occupancy = occupancy.copy()
    cx0, cy0, cx1, cy1 = _obb_aabb_bounds(
        lead_cx,
        lead_cy,
        lead_bw,
        lead_lh,
        lead_angle_rad,
        image_size,
        padding=0.0,
    )
    if pre_occupancy[cy0:cy1, cx0:cx1].any():
        return None

    placed = [
        _RaftShipPlacement(
            svg_text=lead_svg_text,
            cx=float(lead_cx),
            cy=float(lead_cy),
            bw=lead_bw,
            lh=lead_lh,
            angle_deg=lead_angle_deg,
            angle_rad=lead_angle_rad,
            class_id=_ship_class_id(
                lead_lh,
                resolution_m,
                class_id,
                size_thresholds,
            ),
            hull_geom=lead_hull_geom,
            hull_fill=lead_hull_fill,
        )
    ]
    _stamp_geometry_occupancy(staged_occupancy, lead_hull_geom)

    prev_hull = lead_hull_geom
    prev_cx = float(lead_cx)
    prev_cy = float(lead_cy)
    base_angle_deg = lead_angle_deg

    for index in range(1, n_ships):
        angle_deg = base_angle_deg + rng.uniform(-0.75, 0.75)
        angle_rad = math.radians(angle_deg)

        if mixed:
            ship_sensor_az = (sensor_az_world_deg - angle_deg) % 360.0
            svg_text_i = _pick_svg(
                svg_metas,
                rng,
                length_range,
                offnadir_deg,
                ship_sensor_az,
                shipgen_kwargs=shipgen_kwargs,
            )
            _cls_name, bw, lh, _lb = _resolve_ship_dimensions(
                svg_text_i,
                resolution_m,
                rng,
                length_range,
                length_exponent,
            )
        else:
            svg_text_i = svg_text_ref
            scale = rng.uniform(0.9, 1.1)
            bw, lh = _scale_ship_pixel_size(bw0, lh0, scale)

        local_hull = _local_hull_geometry(svg_text_i, bw, lh, angle_deg)
        contact = _offshore_contact_candidate(
            prev_hull,
            prev_cx,
            prev_cy,
            local_hull,
            water_nx,
            water_ny,
            tan_x,
            tan_y,
            bw,
            lh,
            angle_rad,
            image_size,
            berth_water_mask,
            pre_occupancy,
        )
        if contact is None:
            break

        cx, cy, hull_geom = contact
        placed.append(
            _RaftShipPlacement(
                svg_text=svg_text_i,
                cx=float(cx),
                cy=float(cy),
                bw=bw,
                lh=lh,
                angle_deg=angle_deg,
                angle_rad=angle_rad,
                class_id=_ship_class_id(
                    lh,
                    resolution_m,
                    class_id,
                    size_thresholds,
                ),
                hull_geom=hull_geom,
                hull_fill=extract_hull_fill(svg_text_i),
            )
        )
        _stamp_geometry_occupancy(staged_occupancy, hull_geom)
        prev_hull = hull_geom
        prev_cx = float(cx)
        prev_cy = float(cy)

    if not placed:
        return None
    return placed, staged_occupancy


def _place_berthed_cluster(
    berth_water_mask: NDArray[np.bool_],
    occupancy: NDArray[np.bool_],
    berth_segments: list[_BerthSegment],
    svg_metas: list[_SvgMeta] | None,
    resolution_m: float,
    rng: random.Random,
    n_ships: int,
    alpha_range: tuple[float, float],
    class_id: int,
    image_size: int,
    background: NDArray[np.uint8],
    length_range: tuple[float, float] | None,
    length_exponent: float,
    size_thresholds: tuple[float, ...] | None,
    mixed: bool,
    berth_stern: bool,
    blur_sigma: float,
    shadow_azimuth_rad: float | None = None,
    shadow_length: float | None = None,
    shadow_alpha: float = 0.0,
    shadow_alpha_scale: float = 1.0,
    offnadir_deg: float = 0.0,
    sensor_az_world_deg: float = 0.0,
    shipgen_kwargs: dict[str, Any] | None = None,
    berth_runs: list[_BerthRun] | None = None,
) -> list[str]:
    labels: list[str] = []
    runs = list(berth_runs) if berth_runs is not None else _build_berth_runs(berth_segments, berth_water_mask)
    if not runs:
        return labels

    rng.shuffle(runs)
    cluster_alpha = rng.uniform(*alpha_range)
    ship_gap = 4.0
    scene_scale = _CLUSTER_SCENE_SUPERSAMPLE

    for run in runs:
        if run.length <= 1e-6:
            continue

        mid_sample = _sample_berth_run(run, run.length / 2.0)
        if mid_sample is None:
            continue
        mid_x, mid_y, mid_tan_x, mid_tan_y, mid_water_nx, mid_water_ny = mid_sample

        if not berth_stern:
            alongside_result = _place_alongside_berthed_run(
                run,
                mid_sample,
                berth_water_mask,
                occupancy,
                svg_metas,
                resolution_m,
                rng,
                n_ships,
                class_id,
                image_size,
                length_range,
                length_exponent,
                size_thresholds,
                mixed,
                offnadir_deg,
                sensor_az_world_deg,
                shipgen_kwargs,
            )
            if alongside_result is None:
                continue

            placed, staged_occupancy = alongside_result
            occupancy[:] = staged_occupancy
            water_tint = _sample_water_tint(background, round(mid_x), round(mid_y))
            rendered_cluster = _render_vector_raft_cluster(
                placed,
                image_size,
                blur_sigma,
                scene_scale,
                join_tolerance=0.75,
                shadow_azimuth_rad=shadow_azimuth_rad,
                shadow_length=shadow_length,
                shadow_alpha=shadow_alpha,
                shadow_alpha_scale=shadow_alpha_scale,
            )
            shadow_alpha_factor = shadow_alpha * shadow_alpha_scale
            if (
                isinstance(rendered_cluster, tuple)
                and len(rendered_cluster) == 2
                and isinstance(rendered_cluster[1], RgbaLayerPatch)
            ):
                shadow_patch, cluster_patch = rendered_cluster
                if shadow_alpha_factor > 0.0 and shadow_patch is not None:
                    _darken_rgba_patch(
                        background,
                        shadow_patch,
                        shadow_alpha_factor,
                        clip_mask=berth_water_mask,
                    )
                _blend_rgba_patch(
                    background,
                    cluster_patch,
                    cluster_alpha,
                    water_tint,
                )
            else:
                shadow_layer: NDArray[np.uint8]
                cluster_layer: NDArray[np.uint8]
                if isinstance(rendered_cluster, tuple):
                    shadow_layer, cluster_layer = rendered_cluster
                else:
                    shadow_layer = np.zeros((image_size, image_size, 4), dtype=np.uint8)
                    cluster_layer = rendered_cluster
                if shadow_alpha_factor > 0.0:
                    _darken_rgba_layer(
                        background,
                        shadow_layer,
                        shadow_alpha_factor,
                        clip_mask=berth_water_mask,
                    )
                _blend_rgba_layer(
                    background,
                    cluster_layer,
                    cluster_alpha,
                    water_tint,
                )

            for ship in placed:
                corners = compute_obb_corners(
                    float(ship.cx),
                    float(ship.cy),
                    float(ship.bw),
                    float(ship.lh),
                    ship.angle_rad,
                )
                labels.append(format_obb_label(ship.class_id, corners, image_size, image_size))
            return labels

        target_ship_count = min(
            n_ships,
            _max_berthed_ships_for_run(
                run.length,
                ship_gap,
                resolution_m,
                length_range,
                berth_stern,
            ),
        )
        if target_ship_count <= 0:
            continue
        mid_stern_dx = -mid_water_nx if berth_stern else mid_tan_x
        mid_stern_dy = -mid_water_ny if berth_stern else mid_tan_y
        mid_angle_deg = _angle_deg_from_stern_direction(mid_stern_dx, mid_stern_dy)
        ship_sensor_az = (sensor_az_world_deg - mid_angle_deg) % 360.0
        canonical_angle_deg = _angle_deg_from_stern_direction(-1.0, 0.0) if berth_stern else 0.0

        if not mixed:
            svg_text_ref = _pick_svg(
                svg_metas,
                rng,
                length_range,
                offnadir_deg,
                ship_sensor_az,
                shipgen_kwargs=shipgen_kwargs,
            )
            _cls0, bw0, lh0, _lb0 = _resolve_ship_dimensions(
                svg_text_ref,
                resolution_m,
                rng,
                length_range,
                length_exponent,
            )

        ship_specs: list[
            tuple[
                str,
                int,
                int,
                BaseGeometry,
                tuple[int, int, int, int],
                float,
                float,
                float,
            ]
        ] = []
        for index in range(target_ship_count):
            if mixed:
                svg_text_i = _pick_svg(
                    svg_metas,
                    rng,
                    length_range,
                    offnadir_deg,
                    ship_sensor_az,
                    shipgen_kwargs=shipgen_kwargs,
                )
                _cls_name, bw, lh, _lb = _resolve_ship_dimensions(
                    svg_text_i,
                    resolution_m,
                    rng,
                    length_range,
                    length_exponent,
                )
            else:
                svg_text_i = svg_text_ref
                if index == 0:
                    bw, lh = bw0, lh0
                else:
                    scale = rng.uniform(0.9, 1.1)
                    bw, lh = _scale_ship_pixel_size(bw0, lh0, scale)

            canonical_hull = _local_hull_geometry(svg_text_i, bw, lh, canonical_angle_deg)
            min_t, max_t = _geometry_projection_extents(canonical_hull, 0.0, 1.0)
            min_n, _max_n = _geometry_projection_extents(canonical_hull, 1.0, 0.0)
            ship_specs.append(
                (
                    svg_text_i,
                    bw,
                    lh,
                    extract_hull_fill(svg_text_i),
                    min_t,
                    max_t,
                    -min_n,
                )
            )

        active_ship_specs = list(ship_specs)
        total_span = sum(max_t - min_t for *_prefix, min_t, max_t, _water_offset in active_ship_specs)
        total_span += max(0, len(active_ship_specs) - 1) * ship_gap
        while active_ship_specs and total_span > run.length:
            active_ship_specs.pop()
            total_span = sum(
                max_t - min_t
                for *_prefix, min_t, max_t, _water_offset in active_ship_specs
            )
            total_span += max(0, len(active_ship_specs) - 1) * ship_gap

        if not active_ship_specs:
            continue

        while active_ship_specs:
            total_span = sum(
                max_t - min_t
                for *_prefix, min_t, max_t, _water_offset in active_ship_specs
            )
            total_span += max(0, len(active_ship_specs) - 1) * ship_gap
            cursor = (run.length - total_span) / 2.0

            staged_occupancy = occupancy.copy()
            placed = []
            valid = True
            failed_index: int | None = None

            for index, (svg_text_i, bw, lh, hull_fill, min_t, max_t, water_offset) in enumerate(active_ship_specs):
                center_s = cursor - min_t
                sampled = _sample_berth_run(run, center_s)
                if sampled is None:
                    valid = False
                    failed_index = index
                    break

                run_x, run_y, tan_x, tan_y, water_nx, water_ny = sampled
                stern_dx = -water_nx if berth_stern else tan_x
                stern_dy = -water_ny if berth_stern else tan_y
                angle_deg = _angle_deg_from_stern_direction(stern_dx, stern_dy)
                angle_rad = math.radians(angle_deg)
                local_hull = _local_hull_geometry(svg_text_i, bw, lh, angle_deg)
                cx = run_x + water_nx * water_offset
                cy = run_y + water_ny * water_offset
                hull_geom = _translate_hull_geometry(local_hull, cx, cy)

                resolved = _resolve_berth_land_intrusion(
                    berth_water_mask,
                    hull_geom,
                    cx,
                    cy,
                    water_nx,
                    water_ny,
                    max_shift_px=max(14.0, water_offset + 2.0),
                )
                if resolved is None:
                    valid = False
                    failed_index = index
                    break

                cx, cy, hull_geom = resolved
                min_x, min_y, max_x, max_y = hull_geom.bounds
                if min_x < 0.0 or min_y < 0.0 or max_x > image_size or max_y > image_size:
                    valid = False
                    failed_index = index
                    break

                if not _obb_on_berth_water(
                    berth_water_mask,
                    cx,
                    cy,
                    bw,
                    lh,
                    angle_rad,
                    water_nx,
                    water_ny,
                ):
                    valid = False
                    failed_index = index
                    break

                cx0, cy0, cx1, cy1 = _obb_aabb_bounds(
                    cx,
                    cy,
                    bw,
                    lh,
                    angle_rad,
                    image_size,
                    padding=0.0,
                )
                if staged_occupancy[cy0:cy1, cx0:cx1].any():
                    valid = False
                    failed_index = index
                    break

                placed.append(
                    _RaftShipPlacement(
                        svg_text=svg_text_i,
                        cx=float(cx),
                        cy=float(cy),
                        bw=bw,
                        lh=lh,
                        angle_deg=angle_deg,
                        angle_rad=angle_rad,
                        class_id=_ship_class_id(
                            lh,
                            resolution_m,
                            class_id,
                            size_thresholds,
                        ),
                        hull_geom=hull_geom,
                        hull_fill=hull_fill,
                    )
                )
                _stamp_geometry_occupancy(staged_occupancy, hull_geom)
                cursor = center_s + max_t + ship_gap

            if valid and placed:
                occupancy[:] = staged_occupancy
                water_tint = _sample_water_tint(background, round(mid_x), round(mid_y))
                rendered_cluster = _render_vector_raft_cluster(
                    placed,
                    image_size,
                    blur_sigma,
                    scene_scale,
                    join_tolerance=0.0,
                    shadow_azimuth_rad=shadow_azimuth_rad,
                    shadow_length=shadow_length,
                    shadow_alpha=shadow_alpha,
                    shadow_alpha_scale=shadow_alpha_scale,
                )
                shadow_alpha_factor = shadow_alpha * shadow_alpha_scale
                if (
                    isinstance(rendered_cluster, tuple)
                    and len(rendered_cluster) == 2
                    and isinstance(rendered_cluster[1], RgbaLayerPatch)
                ):
                    shadow_patch, cluster_patch = rendered_cluster
                    if shadow_alpha_factor > 0.0 and shadow_patch is not None:
                        _darken_rgba_patch(
                            background,
                            shadow_patch,
                            shadow_alpha_factor,
                            clip_mask=berth_water_mask,
                        )
                    _blend_rgba_patch(
                        background,
                        cluster_patch,
                        cluster_alpha,
                        water_tint,
                    )
                else:
                    shadow_layer: NDArray[np.uint8]
                    cluster_layer: NDArray[np.uint8]
                    if isinstance(rendered_cluster, tuple):
                        shadow_layer, cluster_layer = rendered_cluster
                    else:
                        shadow_layer = np.zeros((image_size, image_size, 4), dtype=np.uint8)
                        cluster_layer = rendered_cluster
                    if shadow_alpha_factor > 0.0:
                        _darken_rgba_layer(
                            background,
                            shadow_layer,
                            shadow_alpha_factor,
                            clip_mask=berth_water_mask,
                        )
                    _blend_rgba_layer(
                        background,
                        cluster_layer,
                        cluster_alpha,
                        water_tint,
                    )

                for ship in placed:
                    corners = compute_obb_corners(
                        float(ship.cx),
                        float(ship.cy),
                        float(ship.bw),
                        float(ship.lh),
                        ship.angle_rad,
                    )
                    labels.append(format_obb_label(ship.class_id, corners, image_size, image_size))
                return labels

            if failed_index is None or failed_index <= 0:
                break

            active_ship_specs = active_ship_specs[:failed_index]

    return labels


def _place_area_cluster(
    water_mask: NDArray[np.bool_],
    occupancy: NDArray[np.bool_],
    svg_metas: list[_SvgMeta] | None,
    resolution_m: float,
    rng: random.Random,
    n_ships: int,
    blur_sigma: float,
    alpha_range: tuple[float, float],
    class_id: int,
    image_size: int,
    background: NDArray[np.uint8],
    length_range: tuple[float, float] | None,
    length_exponent: float,
    size_thresholds: tuple[float, ...] | None,
    mixed: bool,
    shadow_azimuth_rad: float | None = None,
    shadow_length: float | None = None,
    shadow_alpha: float = 0.0,
    shadow_alpha_scale: float = 1.0,
    offnadir_deg: float = 0.0,
    sensor_az_world_deg: float = 0.0,
    shipgen_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Place ships in a loose 2D area with fully random headings."""
    labels: list[str] = []

    # Pre-compute i==0 angle so the reference SVG gets a consistent sensor_az.
    ref_angle_deg = rng.uniform(0, 360)
    ref_sensor_az_ship_deg = (sensor_az_world_deg - ref_angle_deg) % 360.0
    svg_text_ref = _pick_svg(
        svg_metas, rng, length_range, offnadir_deg, ref_sensor_az_ship_deg,
        shipgen_kwargs=shipgen_kwargs,
    )
    cls0, bw0, lh0, _ = _resolve_ship_dimensions(
        svg_text_ref,
        resolution_m,
        rng,
        length_range,
        length_exponent,
    )

    area_radius = max(lh0, int(max(bw0, lh0) * math.sqrt(n_ships) * 0.8))

    available = water_mask & ~occupancy
    pos = find_water_position(available, area_radius * 2, area_radius * 2, 0.0, rng)
    if pos is None:
        return labels

    area_cx, area_cy = pos
    cluster_alpha = rng.uniform(*alpha_range)
    water_tint = _sample_water_tint(background, area_cx, area_cy)

    scene_scale = _CLUSTER_SCENE_SUPERSAMPLE
    scene_size = image_size * scene_scale
    resample_pad = scene_scale * _CLUSTER_RESAMPLE_PAD_OUTPUT_PX
    cluster_items: list[tuple[NDArray[np.uint8], int, int]] = []
    shadow_items: list[tuple[NDArray[np.uint8], int, int]] = []
    placed: list[tuple[float, float, int, int, float, int]] = []

    for i in range(n_ships):
        if i == 0:
            angle_deg = ref_angle_deg
        else:
            angle_deg = rng.uniform(0, 360)
        angle_rad = math.radians(angle_deg)

        if i == 0:
            _cls_name, bw, lh, _lb = _resolve_ship_dimensions(
                svg_text_ref,
                resolution_m,
                rng,
                length_range,
                length_exponent,
            )
            rotated = _rasterize_ship_scene(
                svg_text_ref,
                bw,
                lh,
                angle_deg=angle_deg,
                blur_sigma=blur_sigma,
                scene_scale=scene_scale,
            )
        elif mixed:
            ship_sensor_az = (sensor_az_world_deg - angle_deg) % 360.0
            svg_text_i = _pick_svg(
                svg_metas, rng, length_range, offnadir_deg, ship_sensor_az,
                shipgen_kwargs=shipgen_kwargs,
            )
            _cls_name, bw, lh, _lb = _resolve_ship_dimensions(
                svg_text_i,
                resolution_m,
                rng,
                length_range,
                length_exponent,
            )
            rotated = _rasterize_ship_scene(
                svg_text_i,
                bw,
                lh,
                angle_deg=angle_deg,
                blur_sigma=blur_sigma,
                scene_scale=scene_scale,
            )
        else:
            scale = rng.uniform(0.9, 1.1)
            jit_bw, jit_lh = _scale_ship_pixel_size(bw0, lh0, scale)
            rotated = _rasterize_ship_scene(
                svg_text_ref,
                jit_bw,
                jit_lh,
                angle_deg=angle_deg,
                blur_sigma=blur_sigma,
                scene_scale=scene_scale,
            )
            bw, lh = jit_bw, jit_lh

        for _ in range(60):
            radius = area_radius * math.sqrt(rng.random())
            theta = rng.uniform(0.0, 2.0 * math.pi)
            cx = area_cx + radius * math.cos(theta)
            cy = area_cy + radius * math.sin(theta)

            rh, rw = rotated.shape[:2]
            x0_scene, y0_scene = _cluster_scene_origin(cx, cy, rotated, scene_scale)
            if (
                x0_scene < 0 or x0_scene + rw > scene_size
                or y0_scene < 0 or y0_scene + rh > scene_size
            ):
                continue
            if not _obb_on_water(water_mask, cx, cy, bw, lh, angle_rad):
                continue

            cx0, cy0, cx1, cy1 = _obb_aabb_bounds(
                cx, cy, bw, lh, angle_rad, image_size,
            )
            if occupancy[cy0:cy1, cx0:cx1].any():
                continue

            if (
                shadow_alpha > 0.0
                and shadow_alpha_scale > 0.0
                and shadow_azimuth_rad is not None
                and shadow_length is not None
            ):
                offset_x, offset_y = _shadow_offset_pixels(
                    bw,
                    lh,
                    shadow_azimuth_rad,
                    shadow_length,
                    scene_scale=scene_scale,
                )
                cast_length = math.hypot(offset_x, offset_y)
                shadow_rgba = _make_shadow_rgba(
                    rotated,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    blur_sigma=_shadow_blur_sigma(
                        bw,
                        lh,
                        cast_length,
                        scene_scale=scene_scale,
                    ),
                    alpha_scale=_shadow_alpha_for_ship(bw, lh),
                )
                shadow_x0, shadow_y0 = _cluster_scene_origin(cx, cy, shadow_rgba, scene_scale)
                shadow_items.append((shadow_rgba, shadow_x0, shadow_y0))

            cluster_items.append((rotated, x0_scene, y0_scene))
            cid = _ship_class_id(lh, resolution_m, class_id, size_thresholds)
            placed.append((cx, cy, bw, lh, angle_rad, cid))
            _stamp_occupancy(occupancy, cx, cy, bw, lh, angle_rad)
            break

    if placed:
        shadow_alpha_factor = shadow_alpha * shadow_alpha_scale
        if shadow_alpha_factor > 0.0:
            shadow_patch = _composite_items_to_patch(
                shadow_items,
                scene_size,
                scene_scale,
                padding=resample_pad,
            )
            if shadow_patch is not None:
                _darken_rgba_patch(
                    background,
                    shadow_patch,
                    shadow_alpha_factor,
                    clip_mask=water_mask,
                )
        cluster_patch = _composite_items_to_patch(
            cluster_items,
            scene_size,
            scene_scale,
            padding=resample_pad,
        )
        if cluster_patch is not None:
            _blend_rgba_patch(
                background,
                cluster_patch,
                cluster_alpha,
                water_tint,
            )
        for cx, cy, bw, lh, angle_rad, cid in placed:
            corners = compute_obb_corners(
                float(cx), float(cy), float(bw), float(lh), angle_rad,
            )
            labels.append(format_obb_label(cid, corners, image_size, image_size))

    return labels


def _place_cluster(
    water_mask: NDArray[np.bool_],
    occupancy: NDArray[np.bool_],
    svg_metas: list[_SvgMeta] | None,
    resolution_m: float,
    rng: random.Random,
    cluster_size_range: tuple[int, int],
    blur_sigma: float,
    alpha_range: tuple[float, float],
    class_id: int,
    image_size: int,
    background: NDArray[np.uint8],
    length_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
    size_thresholds: tuple[float, ...] | None = None,
    mixed_prob: float = 0.5,
    raft_open_contact_prob: float = 0.5,
    berth_prob: float = 0.25,
    berth_stern_prob: float = 0.5,
    berth_water_mask: NDArray[np.bool_] | None = None,
    berth_segments: list[_BerthSegment] | None = None,
    berth_runs: list[_BerthRun] | None = None,
    shadow_azimuth_rad: float | None = None,
    shadow_length: float | None = None,
    shadow_alpha: float = 0.0,
    shadow_alpha_scale: float = 1.0,
    offnadir_deg: float = 0.0,
    sensor_az_world_deg: float = 0.0,
    shipgen_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Place a cluster of ships and return label lines."""
    n_ships = rng.randint(*cluster_size_range)
    labels: list[str] = []
    mixed = rng.random() < mixed_prob

    if (
        berth_water_mask is not None
        and berth_segments
        and rng.random() < max(0.0, min(1.0, berth_prob))
    ):
        berthed_labels = _place_berthed_cluster(
            berth_water_mask,
            occupancy,
            berth_segments,
            svg_metas,
            resolution_m,
            rng,
            n_ships,
            alpha_range,
            class_id,
            image_size,
            background,
            length_range,
            length_exponent,
            size_thresholds,
            mixed,
            berth_stern=rng.random() < max(0.0, min(1.0, berth_stern_prob)),
            blur_sigma=blur_sigma,
            shadow_azimuth_rad=shadow_azimuth_rad,
            shadow_length=shadow_length,
            shadow_alpha=shadow_alpha,
            shadow_alpha_scale=shadow_alpha_scale,
            offnadir_deg=offnadir_deg,
            sensor_az_world_deg=sensor_az_world_deg,
            shipgen_kwargs=shipgen_kwargs,
            berth_runs=berth_runs,
        )
        if berthed_labels:
            return berthed_labels

    layout = rng.choices(
        ["raft_tight", "raft_open", "area_scattered"],
        weights=[0.35, 0.30, 0.35],
    )[0]

    if layout == "area_scattered":
        return _place_area_cluster(
            water_mask,
            occupancy,
            svg_metas,
            resolution_m,
            rng,
            n_ships,
            blur_sigma,
            alpha_range,
            class_id,
            image_size,
            background,
            length_range,
            length_exponent,
            size_thresholds,
            mixed,
            shadow_azimuth_rad=shadow_azimuth_rad,
            shadow_length=shadow_length,
            shadow_alpha=shadow_alpha,
            shadow_alpha_scale=shadow_alpha_scale,
            offnadir_deg=offnadir_deg,
            sensor_az_world_deg=sensor_az_world_deg,
            shipgen_kwargs=shipgen_kwargs,
        )

    tight = layout == "raft_tight"
    raft_open_contact_heavy = (
        layout == "raft_open"
        and rng.random() < max(0.0, min(1.0, raft_open_contact_prob))
    )
    heading_jitter = 0.75 if tight else 20.0
    stagger_frac = 0.0 if tight else 0.15
    tight_stagger_mode = tight and rng.random() < 0.30

    base_angle = rng.uniform(0, 360)
    base_angle_rad = math.radians(base_angle)
    cos_base = math.cos(base_angle_rad)
    sin_base = math.sin(base_angle_rad)
    scene_scale = _CLUSTER_SCENE_SUPERSAMPLE

    initial_sensor_az = (sensor_az_world_deg - base_angle) % 360.0
    svg_text = _pick_svg(
        svg_metas, rng, length_range, offnadir_deg, initial_sensor_az,
        shipgen_kwargs=shipgen_kwargs,
    )
    _cls0, bw0, lh0, _lb0 = _resolve_ship_dimensions(
        svg_text,
        resolution_m,
        rng,
        length_range,
        length_exponent,
    )

    available = water_mask & ~occupancy
    pos = find_water_position(available, bw0 * 2, lh0 * 2, base_angle_rad, rng)
    if pos is None:
        return labels

    base_cx, base_cy = pos
    cluster_alpha = rng.uniform(*alpha_range)
    water_tint = _sample_water_tint(background, base_cx, base_cy)
    placed: list[_RaftShipPlacement] = []

    # Cumulative longitudinal drift for stagger mode: sampled once, applied per-ship step
    stagger_step = rng.uniform(-lh0 * 0.20, lh0 * 0.20) if tight_stagger_mode else 0.0
    cumulative_stagger = 0.0

    pre_occupancy = occupancy.copy()
    cursor_edge = 0.0
    prev_row_offset = 0.0
    prev_max_proj = 0.0
    prev_proj_half = 0.0
    prev_hull: BaseGeometry | None = None

    def _raft_open_contact_staggers(initial_stagger: float) -> list[float]:
        candidates = [initial_stagger]
        for candidate in (initial_stagger * 0.5, 0.0):
            if all(abs(candidate - existing) > 1e-6 for existing in candidates):
                candidates.append(candidate)
        return candidates

    def _candidate_geometry(
        local_hull: BaseGeometry,
        row_offset: float,
        longitudinal_offset: float,
    ) -> tuple[float, float, BaseGeometry]:
        cx = base_cx + row_offset * cos_base + longitudinal_offset * (-sin_base)
        cy = base_cy + row_offset * sin_base + longitudinal_offset * cos_base
        return cx, cy, _translate_hull_geometry(local_hull, cx, cy)

    def _candidate_center(
        row_offset: float,
        longitudinal_offset: float,
    ) -> tuple[float, float]:
        return (
            base_cx + row_offset * cos_base + longitudinal_offset * (-sin_base),
            base_cy + row_offset * sin_base + longitudinal_offset * cos_base,
        )

    def _candidate_in_bounds(hull_geom: BaseGeometry) -> bool:
        min_x, min_y, max_x, max_y = hull_geom.bounds
        return min_x >= 0.0 and min_y >= 0.0 and max_x <= image_size and max_y <= image_size

    def _candidate_valid(
        cx: float,
        cy: float,
        hull_geom: BaseGeometry,
        bw: int,
        lh: int,
        angle_rad: float,
    ) -> bool:
        if not _candidate_in_bounds(hull_geom):
            return False
        if not _obb_on_water(water_mask, cx, cy, bw, lh, angle_rad):
            return False
        cx0, cy0, cx1, cy1 = _obb_aabb_bounds(
            cx, cy, bw, lh, angle_rad, image_size, padding=0.0,
        )
        return not pre_occupancy[cy0:cy1, cx0:cx1].any()

    def _contact_candidate(
        local_hull: BaseGeometry,
        base_contact_row: float,
        longitudinal_offset: float,
        bw: int,
        lh: int,
        angle_rad: float,
        min_proj: float,
        proj_half: float,
        *,
        strict: bool,
    ) -> tuple[float, float, float, BaseGeometry] | None:
        if prev_hull is None:
            return None
        obb_penetration_limit = max(1.0, min(prev_proj_half, proj_half) * 0.22)
        local_min_x, local_min_y, local_max_x, local_max_y = map(float, local_hull.bounds)
        samples: list[tuple[float, float, float, float, BaseGeometry]] = []

        def _sample_contact(row_offset: float) -> tuple[float, float, float, BaseGeometry] | None:
            penetration = prev_row_offset + prev_max_proj - (row_offset + min_proj)
            obb_penetration = prev_row_offset + prev_proj_half - (row_offset - proj_half)
            if penetration > 1.0 or obb_penetration > obb_penetration_limit:
                return None

            cx, cy = _candidate_center(row_offset, longitudinal_offset)
            if (
                local_min_x + cx < 0.0
                or local_min_y + cy < 0.0
                or local_max_x + cx > image_size
                or local_max_y + cy > image_size
            ):
                return None

            if strict:
                if not _obb_on_water(water_mask, cx, cy, bw, lh, angle_rad):
                    return None
                cx0, cy0, cx1, cy1 = _obb_aabb_bounds(
                    cx, cy, bw, lh, angle_rad, image_size, padding=0.0,
                )
                if pre_occupancy[cy0:cy1, cx0:cx1].any():
                    return None

            hull_geom = _translate_hull_geometry(local_hull, cx, cy)
            signed_gap = _signed_geometry_gap(prev_hull, hull_geom)
            return signed_gap, cx, cy, hull_geom

        for adjust_steps in range(-2 * scene_scale, 7 * scene_scale + 1):
            row_offset = base_contact_row + adjust_steps / scene_scale
            sampled = _sample_contact(row_offset)
            if sampled is None:
                continue
            signed_gap, cx, cy, hull_geom = sampled
            samples.append((row_offset, signed_gap, cx, cy, hull_geom))

        if not samples:
            return None

        samples.sort(key=lambda item: item[0])
        bracket: tuple[
            tuple[float, float, float, float, BaseGeometry],
            tuple[float, float, float, float, BaseGeometry],
        ] | None = None
        for left, right in zip(samples, samples[1:]):
            if left[1] <= 0.0 <= right[1]:
                bracket = (left, right)
                break
            if right[1] <= 0.0 <= left[1]:
                bracket = (right, left)
                break

        if bracket is not None:
            best = min(
                bracket,
                key=lambda item: (item[1] > 0.0, abs(item[1])),
            )
            low_row, _low_gap, _low_cx, _low_cy, _low_geom = bracket[0]
            high_row, _high_gap, _high_cx, _high_cy, _high_geom = bracket[1]
            for _ in range(12):
                mid_row = (low_row + high_row) / 2.0
                sampled = _sample_contact(mid_row)
                if sampled is None:
                    break
                mid_gap, mid_cx, mid_cy, mid_geom = sampled
                candidate = (mid_row, mid_gap, mid_cx, mid_cy, mid_geom)
                if (mid_gap <= 0.0 and abs(mid_gap) < abs(best[1])) or (
                    best[1] > 0.0 and abs(mid_gap) < abs(best[1])
                ):
                    best = candidate
                if mid_gap <= 0.0:
                    low_row = mid_row
                else:
                    high_row = mid_row
            return best[0], best[2], best[3], best[4]

        row_offset, _signed_gap, cx, cy, hull_geom = min(
            samples,
            key=lambda item: (item[1] > 0.0, abs(item[1]), abs(item[0] - base_contact_row)),
        )
        return row_offset, cx, cy, hull_geom

    for i in range(n_ships):
        angle_deg = base_angle if tight and i == 0 else (
            base_angle + rng.uniform(-heading_jitter, heading_jitter)
        )
        angle_rad = math.radians(angle_deg)

        if i == 0:
            svg_text_i = svg_text
            _cls_name, bw, lh, _lb = _resolve_ship_dimensions(
                svg_text_i,
                resolution_m,
                rng,
                length_range,
                length_exponent,
            )
        elif mixed:
            ship_sensor_az = (sensor_az_world_deg - angle_deg) % 360.0
            svg_text_i = _pick_svg(
                svg_metas, rng, length_range, offnadir_deg, ship_sensor_az,
                shipgen_kwargs=shipgen_kwargs,
            )
            _cls_name, bw, lh, _lb = _resolve_ship_dimensions(
                svg_text_i,
                resolution_m,
                rng,
                length_range,
                length_exponent,
            )
        else:
            svg_text_i = svg_text
            scale = rng.uniform(0.9, 1.1)
            bw, lh = _scale_ship_pixel_size(bw0, lh0, scale)

        local_hull = _local_hull_geometry(svg_text_i, bw, lh, angle_deg)
        min_proj, max_proj = _geometry_projection_extents(local_hull, cos_base, sin_base)

        jitter_rad = abs(angle_rad - base_angle_rad)
        proj_half = (bw * math.cos(jitter_rad) + lh * math.sin(jitter_rad)) / 2.0
        hull_fill = extract_hull_fill(svg_text_i)

        if i == 0:
            row_offset = -min_proj
            cx, cy, hull_geom = _candidate_geometry(local_hull, row_offset, 0.0)
            if not _candidate_valid(cx, cy, hull_geom, bw, lh, angle_rad):
                break
        elif tight:
            cumulative_stagger += stagger_step
            base_contact_row = prev_row_offset + prev_max_proj - min_proj
            contact = _contact_candidate(
                local_hull,
                base_contact_row,
                cumulative_stagger,
                bw,
                lh,
                angle_rad,
                min_proj,
                proj_half,
                strict=True,
            )
            if contact is None:
                break
            row_offset, cx, cy, hull_geom = contact
        else:
            stagger_px = rng.uniform(-lh * stagger_frac, lh * stagger_frac)
            base_contact_row = cursor_edge - min_proj

            if raft_open_contact_heavy:
                contact_choice: tuple[float, float, float, float, BaseGeometry] | None = None
                for candidate_stagger in _raft_open_contact_staggers(stagger_px):
                    if prev_hull is None:
                        break
                    contact = _contact_candidate(
                        local_hull,
                        base_contact_row,
                        candidate_stagger,
                        bw,
                        lh,
                        angle_rad,
                        min_proj,
                        proj_half,
                        strict=False,
                    )
                    if contact is None:
                        continue
                    candidate_row, candidate_cx, candidate_cy, candidate_hull = contact
                    if not _candidate_valid(candidate_cx, candidate_cy, candidate_hull, bw, lh, angle_rad):
                        continue
                    contact_choice = (
                        candidate_row,
                        candidate_stagger,
                        candidate_cx,
                        candidate_cy,
                        candidate_hull,
                    )
                    break

                if contact_choice is not None:
                    row_offset, stagger_px, cx, cy, hull_geom = contact_choice
                else:
                    row_offset = base_contact_row
                    cx, cy, hull_geom = _candidate_geometry(local_hull, row_offset, stagger_px)
                    if not _candidate_valid(cx, cy, hull_geom, bw, lh, angle_rad):
                        cursor_edge = row_offset + max_proj
                        continue
            else:
                gap_mode = rng.random()
                if gap_mode < 1 / 3:
                    gap_px = 0.0
                elif gap_mode < 2 / 3:
                    gap_px = rng.uniform(0.0, 1.0)
                else:
                    gap_px = rng.uniform(bw * 0.2, bw * 0.8)

                if prev_hull is not None and gap_px <= 1.0:
                    contact = _contact_candidate(
                        local_hull,
                        base_contact_row,
                        stagger_px,
                        bw,
                        lh,
                        angle_rad,
                        min_proj,
                        proj_half,
                        strict=False,
                    )
                    row_offset = (contact[0] if contact is not None else base_contact_row) + gap_px
                else:
                    row_offset = base_contact_row + gap_px

                cx, cy, hull_geom = _candidate_geometry(local_hull, row_offset, stagger_px)
                if not _candidate_valid(cx, cy, hull_geom, bw, lh, angle_rad):
                    cursor_edge = row_offset + max_proj
                    continue

        cid = _ship_class_id(lh, resolution_m, class_id, size_thresholds, is_cluster=tight)
        placed.append(
            _RaftShipPlacement(
                svg_text=svg_text_i,
                cx=float(cx),
                cy=float(cy),
                bw=bw,
                lh=lh,
                angle_deg=angle_deg,
                angle_rad=angle_rad,
                class_id=cid,
                hull_geom=hull_geom,
                hull_fill=hull_fill,
            )
        )
        _stamp_geometry_occupancy(occupancy, hull_geom)

        prev_hull = hull_geom
        prev_row_offset = row_offset
        prev_max_proj = max_proj
        prev_proj_half = proj_half
        cursor_edge = row_offset + max_proj

    if placed:
        rendered_cluster = _render_vector_raft_cluster(
            placed,
            image_size,
            blur_sigma,
            scene_scale,
            join_tolerance=0.75 if tight else 0.0,
            shadow_azimuth_rad=shadow_azimuth_rad,
            shadow_length=shadow_length,
            shadow_alpha=shadow_alpha,
            shadow_alpha_scale=shadow_alpha_scale,
        )
        shadow_alpha_factor = shadow_alpha * shadow_alpha_scale
        if (
            isinstance(rendered_cluster, tuple)
            and len(rendered_cluster) == 2
            and isinstance(rendered_cluster[1], RgbaLayerPatch)
        ):
            shadow_patch, cluster_patch = rendered_cluster
            if shadow_alpha_factor > 0.0 and shadow_patch is not None:
                _darken_rgba_patch(
                    background,
                    shadow_patch,
                    shadow_alpha_factor,
                    clip_mask=water_mask,
                )
            _blend_rgba_patch(
                background,
                cluster_patch,
                cluster_alpha,
                water_tint,
            )
        else:
            shadow_layer: NDArray[np.uint8]
            cluster_layer: NDArray[np.uint8]
            if isinstance(rendered_cluster, tuple):
                shadow_layer, cluster_layer = rendered_cluster
            else:
                shadow_layer = np.zeros((image_size, image_size, 4), dtype=np.uint8)
                cluster_layer = rendered_cluster
            if shadow_alpha_factor > 0.0:
                _darken_rgba_layer(
                    background,
                    shadow_layer,
                    shadow_alpha_factor,
                    clip_mask=water_mask,
                )
            _blend_rgba_layer(
                background,
                cluster_layer,
                cluster_alpha,
                water_tint,
            )
        for ship in placed:
            corners = compute_obb_corners(
                float(ship.cx),
                float(ship.cy),
                float(ship.bw),
                float(ship.lh),
                ship.angle_rad,
            )
            labels.append(format_obb_label(ship.class_id, corners, image_size, image_size))

    return labels