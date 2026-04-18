"""Spatial placement and cluster layout helpers for synthetic datagen."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from medetect.datagen.obb import compute_obb_corners, format_obb_label
from medetect.datagen.render import extract_hull_fill, extract_hull_polygon, rasterize_ship_svg
from medetect.datagen.scene import (
    _blend_rgba_layer,
    _cluster_scene_origin,
    _composite_rgba,
    _darken_rgba_layer,
    _downsample_cluster_layer,
    _make_shadow_rgba,
    _rasterize_ship_scene,
    _sample_water_tint,
    _shadow_alpha_for_ship,
    _shadow_blur_sigma,
    _shadow_offset_pixels,
)
from medetect.datagen.ship import (
    _SvgMeta,
    _pick_svg,
    _resolve_ship_dimensions,
    _ship_class_id,
)
from medetect.datagen.svg import parse_svg_metadata

_CLUSTER_SCENE_SUPERSAMPLE: int = 4


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


def _local_hull_geometry(
    svg_text: str,
    beam_px: int,
    length_px: int,
    angle_deg: float,
) -> BaseGeometry:
    """Return the ship hull geometry in pixel units centred at the origin."""
    _ship_class, lb_ratio = parse_svg_metadata(svg_text)
    hull_points = extract_hull_polygon(svg_text)
    sy = float(length_px) / max(lb_ratio, 1e-6)
    pts = [
        ((x - 0.5) * float(beam_px), (y - lb_ratio / 2.0) * sy)
        for x, y in hull_points
    ]
    geometry = Polygon(pts).buffer(0)
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


def _render_vector_raft_cluster(
    ships: list[_RaftShipPlacement],
    image_size: int,
    blur_sigma: float,
    scene_scale: int,
    join_tolerance: float = 0.0,
    shadow_azimuth_rad: float | None = None,
    shadow_length: float | None = None,
    shadow_alpha_scale: float = 1.0,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Render a raft cluster from vector hulls plus per-ship detail layers."""
    scene_size = image_size * scene_scale
    hull_img = Image.new("RGBA", (scene_size, scene_size), (0, 0, 0, 0))
    if join_tolerance > 0.0 and ships:
        merged_hull: BaseGeometry | None = None
        for ship in ships:
            buffered = ship.hull_geom.buffer(join_tolerance)
            merged_hull = buffered if merged_hull is None else merged_hull.union(buffered)
        if merged_hull is not None and not merged_hull.is_empty:
            avg_fill = tuple(
                round(sum(ship.hull_fill[idx] for ship in ships) / len(ships))
                for idx in range(4)
            )
            scaled_underlay = affinity.scale(
                merged_hull,
                xfact=scene_scale,
                yfact=scene_scale,
                origin=(0.0, 0.0),
            )
            _draw_geometry_fill(hull_img, scaled_underlay, avg_fill)
    for ship in ships:
        scaled_hull = affinity.scale(
            ship.hull_geom,
            xfact=scene_scale,
            yfact=scene_scale,
            origin=(0.0, 0.0),
        )
        _draw_geometry_fill(hull_img, scaled_hull, ship.hull_fill)

    layer = np.array(hull_img, dtype=np.uint8)
    shadow_layer = np.zeros((scene_size, scene_size, 4), dtype=np.uint8)
    if (
        shadow_alpha_scale > 0.0
        and shadow_azimuth_rad is not None
        and shadow_length is not None
    ):
        for ship in ships:
            ship_rgba = _rasterize_ship_scene(
                ship.svg_text,
                ship.bw,
                ship.lh,
                angle_deg=ship.angle_deg,
                blur_sigma=0.0,
                scene_scale=scene_scale,
            )
            offset_x, offset_y = _shadow_offset_pixels(
                ship.bw,
                ship.lh,
                shadow_azimuth_rad,
                shadow_length,
                scene_scale=scene_scale,
            )
            cast_length = math.hypot(offset_x, offset_y)
            shadow_rgba = _make_shadow_rgba(
                ship_rgba,
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
            _composite_rgba(shadow_layer, shadow_rgba, shadow_x0, shadow_y0)

    for ship in ships:
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
        _composite_rgba(layer, detail_rgba, x0_scene, y0_scene)

    if blur_sigma > 0 and ships:
        img = Image.fromarray(layer)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma * scene_scale))
        layer = np.array(img)

    return (
        _downsample_cluster_layer(shadow_layer, image_size, scene_scale),
        _downsample_cluster_layer(layer, image_size, scene_scale),
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
    size_threshold: float | None,
    mixed: bool,
    shadow_azimuth_rad: float | None = None,
    shadow_length: float | None = None,
    shadow_alpha_scale: float = 1.0,
) -> list[str]:
    """Place ships in a loose 2D area with fully random headings."""
    labels: list[str] = []

    svg_text_ref = _pick_svg(svg_metas, rng, length_range)
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
    cluster_buf = np.zeros((scene_size, scene_size, 4), dtype=np.uint8)
    shadow_buf = np.zeros((scene_size, scene_size, 4), dtype=np.uint8)
    placed: list[tuple[float, float, int, int, float, int]] = []

    for i in range(n_ships):
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
            svg_text_i = _pick_svg(svg_metas, rng, length_range)
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
            svg_text_u = _pick_svg(svg_metas, rng, length_range)
            scale = rng.uniform(0.9, 1.1)
            jit_bw = max(2, round(bw0 * scale))
            jit_lh = max(3, round(lh0 * scale))
            rotated = _rasterize_ship_scene(
                svg_text_u,
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
                shadow_alpha_scale > 0.0
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
                _composite_rgba(shadow_buf, shadow_rgba, shadow_x0, shadow_y0)

            _composite_rgba(cluster_buf, rotated, x0_scene, y0_scene)
            cid = _ship_class_id(lh, resolution_m, class_id, size_threshold)
            placed.append((cx, cy, bw, lh, angle_rad, cid))
            _stamp_occupancy(occupancy, cx, cy, bw, lh, angle_rad)
            break

    if placed:
        shadow_layer = _downsample_cluster_layer(shadow_buf, image_size, scene_scale)
        _darken_rgba_layer(
            background,
            shadow_layer,
            cluster_alpha * shadow_alpha_scale,
            clip_mask=water_mask,
        )
        cluster_layer = _downsample_cluster_layer(cluster_buf, image_size, scene_scale)
        _blend_rgba_layer(background, cluster_layer, cluster_alpha, water_tint)
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
    size_threshold: float | None = None,
    mixed_prob: float = 0.5,
    shadow_azimuth_rad: float | None = None,
    shadow_length: float | None = None,
    shadow_alpha_scale: float = 1.0,
) -> list[str]:
    """Place a cluster of ships and return label lines."""
    n_ships = rng.randint(*cluster_size_range)
    labels: list[str] = []
    mixed = rng.random() < mixed_prob

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
            size_threshold,
            mixed,
            shadow_azimuth_rad,
            shadow_length,
            shadow_alpha_scale,
        )

    tight = layout == "raft_tight"
    heading_jitter = 0.75 if tight else 20.0
    stagger_frac = 0.0 if tight else 0.15

    base_angle = rng.uniform(0, 360)
    base_angle_rad = math.radians(base_angle)
    cos_base = math.cos(base_angle_rad)
    sin_base = math.sin(base_angle_rad)
    scene_scale = _CLUSTER_SCENE_SUPERSAMPLE

    svg_text = _pick_svg(svg_metas, rng, length_range)
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

    pre_occupancy = occupancy.copy()
    cursor_edge = 0.0
    prev_row_offset = 0.0
    prev_max_proj = 0.0
    prev_proj_half = 0.0
    prev_hull: BaseGeometry | None = None

    def _candidate_geometry(
        local_hull: BaseGeometry,
        row_offset: float,
        longitudinal_offset: float,
    ) -> tuple[float, float, BaseGeometry]:
        cx = base_cx + row_offset * cos_base + longitudinal_offset * (-sin_base)
        cy = base_cy + row_offset * sin_base + longitudinal_offset * cos_base
        return cx, cy, _translate_hull_geometry(local_hull, cx, cy)

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
        samples: list[tuple[float, float, float, float, BaseGeometry]] = []

        def _sample_contact(row_offset: float) -> tuple[float, float, float, BaseGeometry] | None:
            cx, cy, hull_geom = _candidate_geometry(local_hull, row_offset, longitudinal_offset)
            if strict:
                if not _candidate_valid(cx, cy, hull_geom, bw, lh, angle_rad):
                    return None
            elif not _candidate_in_bounds(hull_geom):
                return None

            penetration = prev_row_offset + prev_max_proj - (row_offset + min_proj)
            obb_penetration = prev_row_offset + prev_proj_half - (row_offset - proj_half)
            if penetration > 1.0 or obb_penetration > obb_penetration_limit:
                return None

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
            svg_text_i = _pick_svg(svg_metas, rng, length_range)
            _cls_name, bw, lh, _lb = _resolve_ship_dimensions(
                svg_text_i,
                resolution_m,
                rng,
                length_range,
                length_exponent,
            )
        else:
            svg_text_i = _pick_svg(svg_metas, rng, length_range)
            scale = rng.uniform(0.9, 1.1)
            bw = max(2, round(bw0 * scale))
            lh = max(3, round(lh0 * scale))

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
            base_contact_row = prev_row_offset + prev_max_proj - min_proj
            contact = _contact_candidate(
                local_hull,
                base_contact_row,
                0.0,
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
            if i == 0:
                gap_px = 0.0
            else:
                gap_mode = rng.random()
                if gap_mode < 1 / 3:
                    gap_px = 0.0
                elif gap_mode < 2 / 3:
                    gap_px = rng.uniform(0.0, 1.0)
                else:
                    gap_px = rng.uniform(bw * 0.2, bw * 0.8)

            stagger_px = rng.uniform(-lh * stagger_frac, lh * stagger_frac)
            base_contact_row = cursor_edge - min_proj
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

        cid = _ship_class_id(lh, resolution_m, class_id, size_threshold)
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
        cluster_layer = _render_vector_raft_cluster(
            placed,
            image_size,
            blur_sigma,
            scene_scale,
            join_tolerance=0.75 if tight else 0.0,
            shadow_azimuth_rad=shadow_azimuth_rad,
            shadow_length=shadow_length,
            shadow_alpha_scale=shadow_alpha_scale,
        )
        shadow_layer: NDArray[np.uint8]
        if isinstance(cluster_layer, tuple):
            shadow_layer, cluster_layer = cluster_layer
        else:
            shadow_layer = np.zeros((image_size, image_size, 4), dtype=np.uint8)
        _darken_rgba_layer(background, shadow_layer, cluster_alpha, clip_mask=water_mask)
        _blend_rgba_layer(background, cluster_layer, cluster_alpha, water_tint)
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