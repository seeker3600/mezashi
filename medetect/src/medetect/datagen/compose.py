"""Synthetic ship detection dataset generator.

Composites SVG ship silhouettes onto Sentinel-2 background imagery to
produce a YOLO OBB (oriented bounding box) training dataset.  Water
regions are identified via the Sentinel-2 L2A Scene Classification Layer
(SCL band, value 6 = water).

Output structure::

    output_dir/
        images/train/  ← PNG tiles
        labels/train/  ← YOLO OBB label files
        dataset.yaml   ← YOLO config
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from PIL import Image, ImageFilter
from rasterio.windows import Window
from tqdm import tqdm

from medetect.datagen.render import parse_svg_metadata, rasterize_ship_svg
from medetect.datagen.water_mask import (
    erode_mask,
    make_water_mask_from_rgb,
    make_water_mask_from_scl,
)

logger = logging.getLogger(__name__)


# ── Real-world ship lengths (metres) ─────────────────────────────────────

SHIP_LENGTHS_M: dict[str, tuple[float, float]] = {
    "patrol": (30.0, 80.0),
    "corvette": (80.0, 110.0),
    "frigate": (110.0, 150.0),
    "destroyer": (150.0, 190.0),
    "destroyer_stealth": (140.0, 180.0),
    "carrier": (260.0, 340.0),
    "amphib_assault": (200.0, 260.0),
    "lst_lpd": (120.0, 200.0),
    "supply": (150.0, 210.0),
    "fishing_squid_jigger": (20.0, 50.0),
    "fishing_trawler": (15.0, 40.0),
    "fishing_purse_seiner": (25.0, 60.0),
    "fishing_longliner": (20.0, 45.0),
}

_DEFAULT_LENGTH_M = (30.0, 100.0)


# ── Pure helpers (easily testable) ────────────────────────────────────────


def compute_ship_pixel_size(
    ship_class: str,
    lb_ratio: float,
    resolution_m: float,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
) -> tuple[int, int]:
    """Compute ship raster size ``(beam_px, length_px)`` for the tile resolution.

    A random length within the real-world range for *ship_class* is chosen,
    then converted to pixels at the given *resolution_m*.

    Parameters
    ----------
    length_range
        Global ``(min_m, max_m)`` clamp applied on top of the per-class range.
        When *None*, only the per-class range is used.
    """
    lo, hi = SHIP_LENGTHS_M.get(ship_class, _DEFAULT_LENGTH_M)
    if length_range is not None:
        lo = max(lo, length_range[0])
        hi = min(hi, length_range[1])
        if lo > hi:
            lo, hi = length_range[0], length_range[1]
    length_m = rng.uniform(lo, hi)
    beam_m = length_m / lb_ratio

    length_px = max(3, round(length_m / resolution_m))
    beam_px = max(2, round(beam_m / resolution_m))
    return beam_px, length_px


def compute_obb_corners(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_rad: float,
) -> list[tuple[float, float]]:
    """Compute oriented bounding-box corners.

    Returns four ``(x, y)`` tuples in clockwise order, in pixel
    coordinates (not normalised).
    """
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    hw, hh = w / 2, h / 2

    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in corners
    ]


def format_obb_label(
    class_id: int,
    corners: list[tuple[float, float]],
    img_w: int,
    img_h: int,
) -> str:
    """Format one YOLO OBB label line with normalised coordinates."""
    parts = [str(class_id)]
    for x, y in corners:
        parts.append(f"{x / img_w:.6f}")
        parts.append(f"{y / img_h:.6f}")
    return " ".join(parts)


def find_water_position(
    water_mask: NDArray[np.bool_],
    ship_w: int,
    ship_h: int,
    angle_rad: float,
    rng: random.Random,
    *,
    max_attempts: int = 100,
) -> tuple[int, int] | None:
    """Find a valid centre position for a ship on the water mask.

    The rotated bounding box of the ship must lie entirely within water.
    Returns ``(cx, cy)`` or ``None`` if no position could be found.
    """
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


def blend_ship(
    background: NDArray[np.uint8],
    ship_rgba: NDArray[np.uint8],
    cx: int,
    cy: int,
    alpha_factor: float = 0.85,
    water_tint: NDArray[np.float32] | None = None,
) -> None:
    """Alpha-composite *ship_rgba* onto *background* centred at ``(cx, cy)``.

    Modifies *background* in place.
    """
    sh, sw = ship_rgba.shape[:2]
    x0, y0 = cx - sw // 2, cy - sh // 2

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)

    bh, bw = background.shape[:2]
    copy_w = min(sw - src_x0, bw - dst_x0)
    copy_h = min(sh - src_y0, bh - dst_y0)
    if copy_w <= 0 or copy_h <= 0:
        return

    ship_crop = ship_rgba[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    bg_crop = background[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w]

    alpha = (ship_crop[:, :, 3:4].astype(np.float32) / 255.0) * alpha_factor
    ship_rgb = ship_crop[:, :, :3].astype(np.float32)

    # Slightly tint ship colours toward surrounding water for realism
    if water_tint is not None:
        ship_rgb = ship_rgb * 0.82 + water_tint * 0.18

    blended = bg_crop.astype(np.float32) * (1.0 - alpha) + ship_rgb * alpha
    background[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = (
        blended.clip(0, 255).astype(np.uint8)
    )


# ── SVG loading helpers ───────────────────────────────────────────────────


def _load_svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _pick_svg(
    svg_files: list[Path] | None,
    rng: random.Random,
) -> str:
    """Return SVG text — from file list or generated on-the-fly."""
    if svg_files:
        return _load_svg(rng.choice(svg_files))

    from medetect.shipgen.gen import generate_ship_svg, get_ship_classes

    cls = rng.choice(get_ship_classes())
    return generate_ship_svg(cls, rng=rng)


# ── Tile reading ──────────────────────────────────────────────────────────


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
    # (C, H, W) → (H, W, C)
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
        # SCL at 20 m vs visual at 10 m → factor 2
        scale = scl_src.res[0] / 10.0  # Should be ~2.0
        scl_col = int(col / scale)
        scl_row = int(row / scale)
        scl_size = max(1, int(size / scale))

        scl_col = min(scl_col, scl_src.width - scl_size)
        scl_row = min(scl_row, scl_src.height - scl_size)
        scl_col = max(0, scl_col)
        scl_row = max(0, scl_row)

        window = Window(scl_col, scl_row, scl_size, scl_size)
        scl = scl_src.read(1, window=window)

    # Upsample to target_size via nearest-neighbour
    img = Image.fromarray(scl)
    img = img.resize((target_size, target_size), Image.NEAREST)
    return np.array(img)


def _scl_path_for(visual_path: Path) -> Path:
    """Derive SCL file path from a visual TIF path."""
    name = visual_path.name
    scl_name = name.replace("_visual.tif", "_SCL_20m.tif")
    return visual_path.parent / scl_name


# ── Ship rendering pipeline ──────────────────────────────────────────────


def _render_ship(
    svg_text: str,
    resolution_m: float,
    rng: random.Random,
    blur_sigma: float,
    length_range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.uint8], str, int, int, float]:
    """Render one ship and return ``(rgba, class_name, beam_px, length_px, lb_ratio)``."""
    ship_class, lb_ratio = parse_svg_metadata(svg_text)
    beam_px, length_px = compute_ship_pixel_size(
        ship_class, lb_ratio, resolution_m, rng, length_range,
    )

    rgba = rasterize_ship_svg(svg_text, beam_px, length_px)

    # Gaussian blur to simulate satellite PSF
    if blur_sigma > 0 and min(beam_px, length_px) > 2:
        img = Image.fromarray(rgba)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        rgba = np.array(img)

    return rgba, ship_class, beam_px, length_px, lb_ratio


def _rotate_ship(
    rgba: NDArray[np.uint8],
    angle_deg: float,
) -> NDArray[np.uint8]:
    """Rotate ship RGBA image by *angle_deg* with transparent background."""
    img = Image.fromarray(rgba)
    rotated = img.rotate(-angle_deg, resample=Image.BILINEAR, expand=True)
    return np.array(rotated)


# ── Cluster logic ─────────────────────────────────────────────────────────


def _place_cluster(
    water_mask: NDArray[np.bool_],
    svg_files: list[Path] | None,
    resolution_m: float,
    rng: random.Random,
    cluster_size_range: tuple[int, int],
    blur_sigma: float,
    alpha_range: tuple[float, float],
    class_id: int,
    image_size: int,
    background: NDArray[np.uint8],
    length_range: tuple[float, float] | None = None,
) -> list[str]:
    """Place a cluster of ships side-by-side.  Returns label lines."""
    n_ships = rng.randint(*cluster_size_range)
    base_angle = rng.uniform(0, 360)
    labels: list[str] = []

    # Find a base position
    svg0 = _pick_svg(svg_files, rng)
    rgba0, cls0, bw0, lh0, lb0 = _render_ship(svg0, resolution_m, rng, blur_sigma, length_range)
    angle0_rad = math.radians(base_angle)
    pos = find_water_position(water_mask, bw0 * 2, lh0 * 2, angle0_rad, rng)
    if pos is None:
        return labels

    base_cx, base_cy = pos

    for i in range(n_ships):
        svg = _pick_svg(svg_files, rng) if i > 0 else svg0
        rgba, cls_name, bw, lh, lb = (
            _render_ship(svg, resolution_m, rng, blur_sigma, length_range) if i > 0
            else (rgba0, cls0, bw0, lh0, lb0)
        )
        # Small angle jitter within the cluster
        angle_deg = base_angle + rng.uniform(-10, 10)
        angle_rad = math.radians(angle_deg)

        rotated = _rotate_ship(rgba, angle_deg)

        # Offset perpendicular to heading for side-by-side placement
        perp_angle = angle_rad + math.pi / 2
        offset = i * (bw + rng.randint(0, max(1, bw // 3)))
        cx = base_cx + int(offset * math.cos(perp_angle))
        cy = base_cy + int(offset * math.sin(perp_angle))

        # Boundary check
        rh, rw = rotated.shape[:2]
        if (cx - rw // 2 < 0 or cx + rw // 2 >= image_size
                or cy - rh // 2 < 0 or cy + rh // 2 >= image_size):
            continue

        alpha = rng.uniform(*alpha_range)
        # Water colour tinting from surrounding pixels
        water_tint = _sample_water_tint(background, cx, cy)
        blend_ship(background, rotated, cx, cy, alpha, water_tint)

        corners = compute_obb_corners(float(cx), float(cy), float(bw), float(lh), angle_rad)
        labels.append(format_obb_label(class_id, corners, image_size, image_size))

    return labels


def _sample_water_tint(
    background: NDArray[np.uint8],
    cx: int,
    cy: int,
    radius: int = 8,
) -> NDArray[np.float32]:
    """Sample average water colour around a point for blending tint."""
    h, w = background.shape[:2]
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius)
    region = background[y0:y1, x0:x1]
    if region.size == 0:
        return np.array([40.0, 50.0, 60.0], dtype=np.float32)
    return region.mean(axis=(0, 1)).astype(np.float32)


# ── Main entry point ──────────────────────────────────────────────────────


def generate_dataset(
    bg_dir: Path | str,
    output_dir: Path | str,
    count: int,
    *,
    ship_dir: Path | str | None = None,
    image_size: int = 640,
    resolution: float | None = None,
    ships_per_image: tuple[int, int] = (0, 10),
    cluster_prob: float = 0.15,
    cluster_size: tuple[int, int] = (2, 5),
    class_id: int = 0,
    erode_coast: int = 3,
    min_water_ratio: float = 0.3,
    ship_blur_sigma: float = 0.8,
    ship_alpha: tuple[float, float] = (0.7, 0.95),
    ship_length_range: tuple[float, float] | None = None,
    seed: int | None = None,
) -> dict[str, int]:
    """Generate a synthetic ship detection dataset in YOLO OBB format.

    Parameters
    ----------
    bg_dir
        Directory containing Sentinel-2 ``*_visual.tif`` background images.
        Corresponding ``*_SCL_20m.tif`` files, if present, are used for
        water masking.
    output_dir
        Root of the output YOLO dataset.
    count
        Number of training images to generate.
    ship_dir
        Directory of pre-generated SVG ship files.  When *None*, ships are
        generated on-the-fly with :mod:`medetect.shipgen`.
    image_size
        Output tile edge size in pixels.
    resolution
        Target resolution in m/px.  *None* uses the native GeoTIFF resolution.
    ships_per_image
        ``(min, max)`` number of ships per tile.
    cluster_prob
        Probability that a ship group forms a side-by-side cluster.
    cluster_size
        ``(min, max)`` ships in a cluster.
    class_id
        YOLO class ID for all ships.
    erode_coast
        Pixels to erode from the water mask boundary.
    min_water_ratio
        Minimum fraction of water in a tile to be usable.
    ship_blur_sigma
        Gaussian blur sigma applied to rendered ships.
    ship_alpha
        ``(min, max)`` alpha factor range for ship blending.
    ship_length_range
        Global ``(min_m, max_m)`` constraint on ship length in metres.
        Applied on top of the per-class range.  *None* = no global limit.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict[str, int]
        Statistics: ``images``, ``ships``, ``clusters``, ``skipped``.
    """
    bg_dir = Path(bg_dir)
    output_dir = Path(output_dir)

    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Collect inputs
    visual_files = sorted(bg_dir.glob("*_visual.tif"))
    if not visual_files:
        visual_files = sorted(bg_dir.glob("*.tif"))
    if not visual_files:
        msg = f"No TIF files found in {bg_dir}"
        raise FileNotFoundError(msg)

    svg_files: list[Path] | None = None
    if ship_dir is not None:
        svg_files = sorted(Path(ship_dir).glob("*.svg"))
        if not svg_files:
            msg = f"No SVG files found in {ship_dir}"
            raise FileNotFoundError(msg)

    stats = {"images": 0, "ships": 0, "clusters": 0, "skipped": 0}

    for i in tqdm(range(count), desc="Generating dataset"):
        tif_path = rng.choice(visual_files)

        try:
            result = _compose_one(
                tif_path=tif_path,
                svg_files=svg_files,
                image_size=image_size,
                resolution=resolution,
                ships_per_image=ships_per_image,
                cluster_prob=cluster_prob,
                cluster_size=cluster_size,
                class_id=class_id,
                erode_coast=erode_coast,
                min_water_ratio=min_water_ratio,
                ship_blur_sigma=ship_blur_sigma,
                ship_alpha=ship_alpha,
                ship_length_range=ship_length_range,
                rng=rng,
            )
        except Exception:
            logger.warning(
                "Failed to compose %s — skipping", tif_path.name, exc_info=True,
            )
            stats["skipped"] += 1
            continue

        if result is None:
            stats["skipped"] += 1
            continue

        tile, labels, n_clusters = result

        name = f"{i:06d}"
        Image.fromarray(tile).save(img_out / f"{name}.png")
        (lbl_out / f"{name}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""),
            encoding="utf-8",
        )

        stats["images"] += 1
        stats["ships"] += len(labels)
        stats["clusters"] += n_clusters

        if (i + 1) % 100 == 0 or (i + 1) == count:
            logger.info(
                "Progress %d/%d — images=%d ships=%d clusters=%d skipped=%d",
                i + 1, count,
                stats["images"], stats["ships"], stats["clusters"], stats["skipped"],
            )

    # Write dataset YAML
    _write_dataset_yaml(output_dir, class_id)

    logger.info("Dataset complete: %s", stats)
    return stats


# ── Single image composition ──────────────────────────────────────────────


def _compose_one(
    *,
    tif_path: Path,
    svg_files: list[Path] | None,
    image_size: int,
    resolution: float | None,
    ships_per_image: tuple[int, int],
    cluster_prob: float,
    cluster_size: tuple[int, int],
    class_id: int,
    erode_coast: int,
    min_water_ratio: float,
    ship_blur_sigma: float,
    ship_alpha: tuple[float, float],
    ship_length_range: tuple[float, float] | None,
    rng: random.Random,
    max_crop_attempts: int = 20,
) -> tuple[NDArray[np.uint8], list[str], int] | None:
    """Compose one training image.  Returns ``(tile, labels, n_clusters)``."""
    with rasterio.open(tif_path) as src:
        native_res = (src.res[0] + src.res[1]) / 2.0

        # Handle geographic CRS (degree units)
        if src.crs and src.crs.is_geographic:
            center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
            native_res = native_res * 111320.0 * math.cos(math.radians(center_lat))

        if resolution is not None:
            src_tile = max(1, round(image_size * resolution / native_res))
        else:
            src_tile = image_size
            resolution = native_res

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
                    tif_path.name, col, row,
                )
                continue

            # Resize if we read a different size from the output
            if src_tile != image_size:
                img = Image.fromarray(tile)
                img = img.resize((image_size, image_size), Image.BILINEAR)
                tile = np.array(img)

            # Water mask
            scl_file = _scl_path_for(tif_path)
            scl = _read_scl_tile(scl_file, col, row, src_tile, image_size)
            if scl is not None:
                water_mask = make_water_mask_from_scl(scl)
            else:
                water_mask = make_water_mask_from_rgb(tile)

            water_mask = erode_mask(water_mask, erode_coast)

            water_ratio = water_mask.sum() / water_mask.size
            if water_ratio >= min_water_ratio:
                break
        else:
            return None

    # Place ships
    n_ships = rng.randint(*ships_per_image)
    labels: list[str] = []
    n_clusters = 0
    placed = 0

    while placed < n_ships:
        is_cluster = rng.random() < cluster_prob and (n_ships - placed) >= cluster_size[0]

        if is_cluster:
            new_labels = _place_cluster(
                water_mask, svg_files, resolution, rng,
                cluster_size, ship_blur_sigma, ship_alpha,
                class_id, image_size, tile, ship_length_range,
            )
            labels.extend(new_labels)
            placed += max(len(new_labels), cluster_size[0])
            if new_labels:
                n_clusters += 1
        else:
            svg_text = _pick_svg(svg_files, rng)
            rgba, cls_name, bw, lh, lb = _render_ship(
                svg_text, resolution, rng, ship_blur_sigma, ship_length_range,
            )
            angle_deg = rng.uniform(0, 360)
            angle_rad = math.radians(angle_deg)
            rotated = _rotate_ship(rgba, angle_deg)

            pos = find_water_position(water_mask, bw, lh, angle_rad, rng)
            if pos is not None:
                cx, cy = pos
                alpha = rng.uniform(*ship_alpha)
                water_tint = _sample_water_tint(tile, cx, cy)
                blend_ship(tile, rotated, cx, cy, alpha, water_tint)

                corners = compute_obb_corners(
                    float(cx), float(cy), float(bw), float(lh), angle_rad,
                )
                labels.append(
                    format_obb_label(class_id, corners, image_size, image_size),
                )
            placed += 1

    return tile, labels, n_clusters


# ── Dataset YAML ──────────────────────────────────────────────────────────


def _write_dataset_yaml(output_dir: Path, class_id: int) -> None:
    """Write a minimal YOLO dataset YAML config."""
    yaml_path = output_dir / "dataset.yaml"
    content = (
        f"path: {output_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/train\n"
        f"\n"
        f"names:\n"
        f"  {class_id}: ship\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
