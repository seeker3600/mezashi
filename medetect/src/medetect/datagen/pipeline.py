"""Dataset pipeline orchestration and false-negative extraction."""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

from medetect.datagen.scene import DEFAULT_EDGE_HARDNESS
from medetect.datagen.ship import _SvgMeta, _load_svg_metas, _size_class_names
from medetect.datagen.water_mask import CoastlineIndex

logger = logging.getLogger(__name__)

_worker_svg_metas: list[_SvgMeta] | None = None
_worker_coastline_index: CoastlineIndex | None = None
_SURFACE_TARGET_MAX_ATTEMPTS = 24
_SURFACE_TARGET_CROP_ATTEMPTS = 64

_ComposeTaskResult = tuple[int, int, str]


@dataclass(frozen=True)
class _ComposeTaskConfig:
    image_size: int
    resolution: float | None
    geo_scale: float | None
    ships_per_image: tuple[int, int]
    cluster_prob: float
    cluster_size: tuple[int, int]
    cluster_mixed_prob: float
    class_id: int
    erode_coast: int
    min_water_ratio: float
    edge_hardness: float
    ship_alpha: tuple[float, float]
    ship_length_range: tuple[float, float] | None
    length_exponent: float
    berth_prob: float
    berth_stern_prob: float
    size_thresholds: tuple[float, ...] | None
    wake_prob_scale: float
    wake_alpha_scale: float
    debug_bg_color: tuple[int, int, int] | None
    shadow_alpha_scale: float
    shadow_length_range: tuple[float, float]
    offnadir_range: tuple[float, float]
    shipgen_kwargs: dict[str, Any]
    ship_lb_ratio_range: tuple[float, float] | None = None


def _worker_init(
    svg_dir: Path | None,
    coastline_path: Path | None = None,
) -> None:
    """Load SVG metadata and optional coastline index into process-local globals."""
    global _worker_svg_metas, _worker_coastline_index  # noqa: PLW0603
    if svg_dir is not None:
        svg_files = sorted(svg_dir.glob("*.svg"))
        _worker_svg_metas = _load_svg_metas(svg_files)
    else:
        _worker_svg_metas = None

    if coastline_path is not None:
        _worker_coastline_index = CoastlineIndex(coastline_path)
    else:
        _worker_coastline_index = None


def _false_source_grid(
    path: Path,
    image_size: int,
    resolution: float | None,
    geo_scale: float | None,
) -> tuple[int, int, int] | None:
    """Return ``(src_tile, n_cols, n_rows)`` for a false-negative source image."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            with rasterio.open(path) as src:
                width, height = src.width, src.height
                if geo_scale is not None:
                    src_tile = max(1, round(image_size * geo_scale))
                elif resolution is not None:
                    native_res = (src.res[0] + src.res[1]) / 2.0
                    if src.crs and src.crs.is_geographic:
                        center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                        native_res = (
                            native_res
                            * 111320.0
                            * math.cos(math.radians(center_lat))
                        )
                    src_tile = max(1, round(image_size * resolution / native_res))
                else:
                    src_tile = image_size
        else:
            with Image.open(path) as img:
                width, height = img.size
            src_tile = image_size
    except Exception:
        logger.warning(
            "Cannot open false-negative source %s — skipping",
            path.name,
            exc_info=True,
        )
        return None

    n_cols = width // src_tile
    n_rows = height // src_tile
    if n_cols == 0 or n_rows == 0:
        logger.debug(
            "Source %s too small for %d px tiles — skipping",
            path.name,
            image_size,
        )
        return None
    return src_tile, n_cols, n_rows


def _has_black_nodata_pixels(tile: np.ndarray) -> bool:
    """Return True when a tile contains pure-black redaction pixels."""
    return bool(
        np.any(
            (tile[:, :, 0] == 0)
            & (tile[:, :, 1] == 0)
            & (tile[:, :, 2] == 0),
        )
    )


def generate_false_negatives(
    false_dir: Path,
    output_dir: Path,
    count: int,
    image_size: int,
    *,
    resolution: float | None = None,
    geo_scale: float | None = None,
    rng: random.Random,
    start_index: int = 0,
) -> int:
    """Write *count* false-negative (label-free) tiles from *false_dir*."""
    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"

    sources_raw: list[Path] = []
    for pattern in ("*.png", "*.tif", "*.tiff", "*.PNG", "*.TIF", "*.TIFF"):
        sources_raw.extend(false_dir.glob(pattern))
    seen: set[Path] = set()
    sources: list[Path] = []
    for path in sorted(sources_raw):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            sources.append(path)

    if not sources:
        msg = f"No PNG/TIFF images found in {false_dir}"
        raise FileNotFoundError(msg)

    source_grids: list[tuple[Path, int, int, int]] = []
    for path in sources:
        info = _false_source_grid(path, image_size, resolution, geo_scale)
        if info is not None:
            src_tile, cols, rows = info
            source_grids.append((path, src_tile, cols, rows))

    if not source_grids:
        msg = (
            f"No usable source images in {false_dir} "
            f"(all too small for {image_size}px tiles)"
        )
        raise ValueError(msg)

    n_sources = len(source_grids)
    max_per_source = math.ceil(count / n_sources)

    allocations: list[tuple[Path, int, int, int, int]] = []
    remaining = count
    for path, src_tile, cols, rows in source_grids:
        alloc = min(cols * rows, max_per_source, remaining)
        allocations.append((path, src_tile, cols, rows, alloc))
        remaining -= alloc

    if remaining > 0:
        for index, (path, src_tile, cols, rows, alloc) in enumerate(allocations):
            extra = min(cols * rows - alloc, remaining)
            if extra > 0:
                allocations[index] = (path, src_tile, cols, rows, alloc + extra)
                remaining -= extra
            if remaining <= 0:
                break

    if remaining > 0:
        total_cap = sum(cols * rows for _, _, cols, rows in source_grids)
        logger.warning(
            "False-negative sources only provide %d non-overlapping tiles "
            "(requested %d). Tiles will be repeated to reach the target count.",
            total_cap,
            count,
        )
        source_idx = 0
        while remaining > 0:
            path, src_tile, cols, rows, alloc = allocations[source_idx % n_sources]
            allocations[source_idx % n_sources] = (path, src_tile, cols, rows, alloc + 1)
            remaining -= 1
            source_idx += 1

    total_to_write = sum(alloc for *_, alloc in allocations)
    total_written = 0
    idx = start_index
    tiff_suffixes = {".tif", ".tiff"}
    skipped_black_tiles = 0

    def _emit_tile(tile_img: Image.Image) -> None:
        nonlocal idx, total_written
        name = f"{idx:06d}"
        tile_img.save(img_out / f"{name}.png")
        (lbl_out / f"{name}.txt").write_text("", encoding="utf-8")
        idx += 1
        total_written += 1
        pbar.update(1)

    with tqdm(
        total=total_to_write,
        desc="False negatives",
        unit="tile",
        dynamic_ncols=True,
    ) as pbar:
        for path, src_tile, cols, rows, alloc in allocations:
            if alloc <= 0:
                continue

            grid = [(col, row) for row in range(rows) for col in range(cols)]
            rng.shuffle(grid)
            valid_positions: list[tuple[int, int]] = []
            written_for_source = 0

            if path.suffix.lower() in tiff_suffixes:
                with rasterio.open(path) as src:
                    for col, row in grid:
                        x0, y0 = col * src_tile, row * src_tile
                        data = src.read(
                            list(range(1, min(src.count, 3) + 1)),
                            window=Window(x0, y0, src_tile, src_tile),
                        )
                        tile_img = Image.fromarray(
                            np.moveaxis(data, 0, -1).astype(np.uint8)
                        ).convert("RGB")
                        if src_tile != image_size:
                            tile_img = tile_img.resize(
                                (image_size, image_size), Image.BILINEAR,
                            )
                        tile_arr = np.asarray(tile_img, dtype=np.uint8)
                        if _has_black_nodata_pixels(tile_arr):
                            skipped_black_tiles += 1
                            continue
                        valid_positions.append((col, row))
                        _emit_tile(tile_img)
                        written_for_source += 1
                        if written_for_source >= alloc:
                            break

                    if written_for_source < alloc and valid_positions:
                        repeat_positions = list(valid_positions)
                        repeat_index = 0
                        while written_for_source < alloc:
                            if repeat_index > 0 and repeat_index % len(repeat_positions) == 0:
                                rng.shuffle(repeat_positions)
                            col, row = repeat_positions[repeat_index % len(repeat_positions)]
                            repeat_index += 1
                            x0, y0 = col * src_tile, row * src_tile
                            data = src.read(
                                list(range(1, min(src.count, 3) + 1)),
                                window=Window(x0, y0, src_tile, src_tile),
                            )
                            tile_img = Image.fromarray(
                                np.moveaxis(data, 0, -1).astype(np.uint8)
                            ).convert("RGB")
                            if src_tile != image_size:
                                tile_img = tile_img.resize(
                                    (image_size, image_size), Image.BILINEAR,
                                )
                            _emit_tile(tile_img)
                            written_for_source += 1
            else:
                with Image.open(path) as src_img:
                    src_rgb = src_img.convert("RGB")
                    for col, row in grid:
                        x0, y0 = col * src_tile, row * src_tile
                        tile_img = src_rgb.crop((x0, y0, x0 + src_tile, y0 + src_tile))
                        if src_tile != image_size:
                            tile_img = tile_img.resize(
                                (image_size, image_size), Image.BILINEAR,
                            )
                        tile_arr = np.asarray(tile_img, dtype=np.uint8)
                        if _has_black_nodata_pixels(tile_arr):
                            skipped_black_tiles += 1
                            continue
                        valid_positions.append((col, row))
                        _emit_tile(tile_img)
                        written_for_source += 1
                        if written_for_source >= alloc:
                            break

                    if written_for_source < alloc and valid_positions:
                        repeat_positions = list(valid_positions)
                        repeat_index = 0
                        while written_for_source < alloc:
                            if repeat_index > 0 and repeat_index % len(repeat_positions) == 0:
                                rng.shuffle(repeat_positions)
                            col, row = repeat_positions[repeat_index % len(repeat_positions)]
                            repeat_index += 1
                            x0, y0 = col * src_tile, row * src_tile
                            tile_img = src_rgb.crop((x0, y0, x0 + src_tile, y0 + src_tile))
                            if src_tile != image_size:
                                tile_img = tile_img.resize(
                                    (image_size, image_size), Image.BILINEAR,
                                )
                            _emit_tile(tile_img)
                            written_for_source += 1

    if skipped_black_tiles:
        logger.warning(
            "Skipped %d false-negative crop(s) containing blacked-out pixels; wrote %d / %d requested.",
            skipped_black_tiles,
            total_written,
            count,
        )

    logger.info("False negatives written: %d / %d requested", total_written, count)
    return total_written


def generate_dataset(
    bg_dir: Path | str | None,
    output_dir: Path | str,
    count: int,
    *,
    ship_dir: Path | str | None = None,
    image_size: int = 640,
    resolution: float | None = None,
    geo_scale: float | None = None,
    ships_per_image: tuple[int, int] = (0, 10),
    cluster_prob: float = 0.15,
    cluster_size: tuple[int, int] = (2, 5),
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
    seed: int | None = None,
    size_thresholds: tuple[float, ...] | None = None,
    wake_prob_scale: float = 1.0,
    wake_alpha_scale: float = 1.0,
    debug_bg_color: tuple[int, int, int] | None = None,
    shadow_alpha_scale: float = 1.0,
    shadow_length_range: tuple[float, float] = (0.0, 3.75),
    max_workers: int | None = None,
    false_dir: Path | str | None = None,
    false_ratio: float = 0.0,
    bg_surface_mix_ratio: tuple[float, float] | None = None,
    coastline: Path | str | None = None,
    override: bool = False,
    offnadir_range: tuple[float, float] = (0.0, 0.0),
    shipgen_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a synthetic ship detection dataset in YOLO OBB format."""
    bg_dir = Path(bg_dir) if bg_dir is not None else None
    output_dir = Path(output_dir)

    if output_dir.exists():
        if not override:
            msg = f"Output directory {output_dir} already exists. Use --override to overwrite."
            raise FileExistsError(msg)
        else:
            shutil.rmtree(output_dir)
            logger.info("Removed existing output directory %s", output_dir)

    img_out = output_dir / "images" / "train"
    lbl_out = output_dir / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    if false_dir is not None and false_ratio > 0.0:
        if not (0.0 < false_ratio < 1.0):
            msg = f"false_ratio must be in (0, 1), got {false_ratio}"
            raise ValueError(msg)
        false_count = round(count * false_ratio)
        synth_count = count - false_count
    else:
        false_count = 0
        synth_count = count

    if bg_dir is None:
        msg = "bg_dir must be specified"
        raise ValueError(msg)
    visual_files = sorted(bg_dir.glob("*_visual.tif"))
    if not visual_files:
        visual_files = sorted(bg_dir.glob("*.tif"))
    if not visual_files:
        msg = f"No TIF files found in {bg_dir}"
        raise FileNotFoundError(msg)

    svg_dir: Path | None = None
    if ship_dir is not None:
        svg_dir = Path(ship_dir)
        if not any(svg_dir.glob("*.svg")):
            msg = f"No SVG files found in {svg_dir}"
            raise FileNotFoundError(msg)

    coastline_path: Path | None = None
    if coastline is not None:
        coastline_path = Path(coastline)
        if not coastline_path.exists():
            msg = f"Coastline shapefile not found: {coastline_path}"
            raise FileNotFoundError(msg)

    task_tifs = [rng.choice(visual_files) for _ in range(synth_count)]
    task_seeds = [rng.randint(0, 2**32 - 1) for _ in range(synth_count)]

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    elif max_workers < 0:
        msg = f"max_workers must be >= 0, got {max_workers}"
        raise ValueError(msg)

    stats: dict[str, Any] = {
        "images": 0,
        "ships": 0,
        "clusters": 0,
        "skipped": 0,
        "sea_only": 0,
        "mixed": 0,
        "land_only": 0,
        "surface_goal_rejected": 0,
    }

    task_config = _ComposeTaskConfig(
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
        size_thresholds=size_thresholds,
        wake_prob_scale=wake_prob_scale,
        wake_alpha_scale=wake_alpha_scale,
        debug_bg_color=debug_bg_color,
        shadow_alpha_scale=shadow_alpha_scale,
        shadow_length_range=shadow_length_range,
        offnadir_range=offnadir_range,
        shipgen_kwargs=shipgen_kwargs or {},
    )

    def _record_compose_result(n_ships: int, n_clusters: int, surface: str) -> None:
        if n_ships < 0:
            stats["skipped"] += 1
            return
        stats["images"] += 1
        stats["ships"] += n_ships
        stats["clusters"] += n_clusters
        if surface in ("sea_only", "mixed", "land_only"):
            stats[surface] += 1

    expected_surface_by_index: dict[int, str | None] = {}
    if bg_surface_mix_ratio is not None and synth_count > 0:
        sea_ratio, mixed_ratio = bg_surface_mix_ratio
        ratio_sum = sea_ratio + mixed_ratio
        if ratio_sum <= 0.0:
            msg = "bg_surface_mix_ratio must have positive sum"
            raise ValueError(msg)
        sea_target = round(synth_count * sea_ratio / ratio_sum)
        mixed_target = synth_count - sea_target
        targets = ["sea_only"] * sea_target + ["mixed"] * mixed_target
        rng.shuffle(targets)
        expected_surface_by_index = {
            index: targets[index]
            for index in range(synth_count)
        }

    def _expected_surface(index: int) -> str | None:
        return expected_surface_by_index.get(index)

    if max_workers == 0:
        _worker_init(svg_dir, coastline_path)
        with tqdm(
            total=synth_count,
            desc="Generating dataset",
            unit="image",
            dynamic_ncols=True,
        ) as pbar:
            for index, tif_path in enumerate(task_tifs):
                try:
                    result = _run_compose_task(
                        index=index,
                        task_seed=task_seeds[index],
                        tif_path=tif_path,
                        img_out=img_out,
                        lbl_out=lbl_out,
                        config=task_config,
                        expected_surface=_expected_surface(index),
                        candidate_tifs=tuple(visual_files),
                    )
                    n_ships, n_clusters, surface = result
                    if n_ships >= 0 and _expected_surface(index) is not None and surface != _expected_surface(index):
                        stats["surface_goal_rejected"] += 1
                    _record_compose_result(n_ships, n_clusters, surface)
                except Exception:
                    logger.warning(
                        "Failed to compose %s — skipping",
                        tif_path.name,
                        exc_info=True,
                    )
                    stats["skipped"] += 1
                finally:
                    pbar.update(1)
    else:
        max_inflight = max_workers * 2

        task_iter = iter(range(synth_count))
        pending: set[concurrent.futures.Future[_ComposeTaskResult]] = set()
        future_info: dict[
            concurrent.futures.Future[_ComposeTaskResult],
            tuple[int, Path],
        ] = {}

        def _submit_next() -> bool:
            try:
                index = next(task_iter)
            except StopIteration:
                return False
            future = executor.submit(
                _run_compose_task,
                index=index,
                task_seed=task_seeds[index],
                tif_path=task_tifs[index],
                img_out=img_out,
                lbl_out=lbl_out,
                config=task_config,
                expected_surface=_expected_surface(index),
                candidate_tifs=tuple(visual_files),
            )
            pending.add(future)
            future_info[future] = (index, task_tifs[index])
            return True

        def _collect_done(pbar: tqdm) -> None:  # type: ignore[type-arg]
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                pending.discard(future)
                _index, tif_path = future_info.pop(future)
                try:
                    n_ships, n_clusters, surface = future.result()
                    if n_ships >= 0 and _expected_surface(_index) is not None and surface != _expected_surface(_index):
                        stats["surface_goal_rejected"] += 1
                    _record_compose_result(n_ships, n_clusters, surface)
                except Exception:
                    logger.warning(
                        "Failed to compose %s — skipping",
                        tif_path.name,
                        exc_info=True,
                    )
                    stats["skipped"] += 1
                finally:
                    pbar.update(1)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(svg_dir, coastline_path),
        ) as executor:
            with tqdm(
                total=synth_count,
                desc="Generating dataset",
                unit="image",
                dynamic_ncols=True,
            ) as pbar:
                for _ in range(min(max_inflight, synth_count)):
                    _submit_next()

                while pending:
                    _collect_done(pbar)
                    while len(pending) < max_inflight:
                        if not _submit_next():
                            break

    gen_params: dict[str, object] = {
        "count": count,
        "image_size": image_size,
        "resolution": resolution,
        "geo_scale": geo_scale,
        "ships_per_image": f"{ships_per_image[0]}:{ships_per_image[1]}",
        "cluster_prob": cluster_prob,
        "cluster_size": f"{cluster_size[0]}:{cluster_size[1]}",
        "class_id": class_id,
        "erode_coast": erode_coast,
        "min_water_ratio": min_water_ratio,
        "cluster_mixed_prob": cluster_mixed_prob,
        "edge_hardness": edge_hardness,
        "ship_alpha": f"{ship_alpha[0]}:{ship_alpha[1]}",
        "ship_length_range": (
            f"{ship_length_range[0]}:{ship_length_range[1]}"
            if ship_length_range is not None
            else None
        ),
        "ship_lb_ratio_range": (
            f"{ship_lb_ratio_range[0]}:{ship_lb_ratio_range[1]}"
            if ship_lb_ratio_range is not None
            else None
        ),
        "length_exponent": length_exponent,
        "berth_prob": berth_prob,
        "berth_stern_prob": berth_stern_prob,
        "berth_cluster_auto_truncate": True,
        "seed": seed,
        "size_thresholds": (
            list(size_thresholds) if size_thresholds is not None else None
        ),
        "wake_prob_scale": wake_prob_scale,
        "wake_alpha_scale": wake_alpha_scale,
        "shadow_alpha_scale": shadow_alpha_scale,
        "shadow_length_range": f"{shadow_length_range[0]}:{shadow_length_range[1]}",
        "false_dir": str(false_dir) if false_dir is not None else None,
        "false_ratio": false_ratio,
        "bg_surface_mix_ratio": (
            f"{bg_surface_mix_ratio[0]}:{bg_surface_mix_ratio[1]}"
            if bg_surface_mix_ratio is not None
            else None
        ),
        "coastline": str(coastline_path) if coastline_path is not None else None,
    }
    _write_dataset_yaml(
        output_dir,
        class_id,
        size_thresholds=size_thresholds,
        params=gen_params,
    )

    stats["false_negatives"] = 0
    if false_count > 0:
        stats["false_negatives"] = generate_false_negatives(
            false_dir=Path(false_dir),  # type: ignore[arg-type]
            output_dir=output_dir,
            count=false_count,
            image_size=image_size,
            resolution=resolution,
            geo_scale=geo_scale,
            rng=rng,
            start_index=synth_count,
        )

    logger.info("Dataset complete: %s", stats)
    return stats


def _run_compose_task(
    *,
    index: int,
    task_seed: int,
    tif_path: Path | None,
    img_out: Path,
    lbl_out: Path,
    config: _ComposeTaskConfig,
    expected_surface: str | None = None,
    candidate_tifs: tuple[Path, ...] = (),
    surface_target_attempts: int = _SURFACE_TARGET_MAX_ATTEMPTS,
    surface_crop_attempts: int = _SURFACE_TARGET_CROP_ATTEMPTS,
) -> _ComposeTaskResult:
    """Worker function for one dataset image."""
    from medetect.datagen.compose import _compose_one_with_surface_category

    rng = random.Random(task_seed)
    tif_pool = candidate_tifs or ((tif_path,) if tif_path is not None else ())

    def _compose_for_tif(selected_tif: Path | None) -> tuple[int, int, str]:
        if selected_tif is None:
            return -1, -1, "unknown"

        result = _compose_one_with_surface_category(
            tif_path=selected_tif,
            svg_metas=_worker_svg_metas,
            image_size=config.image_size,
            resolution=config.resolution,
            geo_scale=config.geo_scale,
            ships_per_image=config.ships_per_image,
            cluster_prob=config.cluster_prob,
            cluster_size=config.cluster_size,
            cluster_mixed_prob=config.cluster_mixed_prob,
            class_id=config.class_id,
            erode_coast=config.erode_coast,
            min_water_ratio=config.min_water_ratio,
            edge_hardness=config.edge_hardness,
            ship_alpha=config.ship_alpha,
            ship_length_range=config.ship_length_range,
            ship_lb_ratio_range=config.ship_lb_ratio_range,
            length_exponent=config.length_exponent,
            berth_prob=config.berth_prob,
            berth_stern_prob=config.berth_stern_prob,
            rng=rng,
            size_thresholds=config.size_thresholds,
            wake_prob_scale=config.wake_prob_scale,
            wake_alpha_scale=config.wake_alpha_scale,
            debug_bg_color=config.debug_bg_color,
            shadow_alpha_scale=config.shadow_alpha_scale,
            shadow_length_range=config.shadow_length_range,
            coastline_index=_worker_coastline_index,
            offnadir_range=config.offnadir_range,
            shipgen_kwargs=config.shipgen_kwargs,
            required_surface=None,
        )

        if result is None:
            return -1, -1, "unknown"

        tile, labels, n_clusters, surface = result
        name = f"{index:06d}"
        Image.fromarray(tile).save(img_out / f"{name}.png")
        (lbl_out / f"{name}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""),
            encoding="utf-8",
        )
        return len(labels), n_clusters, surface

    def _target_tif_sequence() -> list[Path | None]:
        if not tif_pool:
            return [tif_path]

        ordered_unique: list[Path] = []
        seen: set[Path] = set()
        if tif_path is not None and tif_path in tif_pool:
            ordered_unique.append(tif_path)
            seen.add(tif_path)

        rest = [path for path in tif_pool if path not in seen]
        rng.shuffle(rest)
        ordered_unique.extend(rest)
        return ordered_unique or [tif_path]

    target_tifs = _target_tif_sequence()

    def _sample_tif(attempt: int) -> Path | None:
        if not target_tifs:
            return tif_path
        return target_tifs[attempt % len(target_tifs)]

    if expected_surface is not None:
        for attempt in range(max(1, surface_target_attempts)):
            selected_tif = _sample_tif(attempt)
            if selected_tif is None:
                break
            result = _compose_one_with_surface_category(
                tif_path=selected_tif,
                svg_metas=_worker_svg_metas,
                image_size=config.image_size,
                resolution=config.resolution,
                geo_scale=config.geo_scale,
                ships_per_image=config.ships_per_image,
                cluster_prob=config.cluster_prob,
                cluster_size=config.cluster_size,
                cluster_mixed_prob=config.cluster_mixed_prob,
                class_id=config.class_id,
                erode_coast=config.erode_coast,
                min_water_ratio=config.min_water_ratio,
                edge_hardness=config.edge_hardness,
                ship_alpha=config.ship_alpha,
                ship_length_range=config.ship_length_range,
                ship_lb_ratio_range=config.ship_lb_ratio_range,
                length_exponent=config.length_exponent,
                berth_prob=config.berth_prob,
                berth_stern_prob=config.berth_stern_prob,
                rng=rng,
                size_thresholds=config.size_thresholds,
                wake_prob_scale=config.wake_prob_scale,
                wake_alpha_scale=config.wake_alpha_scale,
                debug_bg_color=config.debug_bg_color,
                shadow_alpha_scale=config.shadow_alpha_scale,
                shadow_length_range=config.shadow_length_range,
                coastline_index=_worker_coastline_index,
                offnadir_range=config.offnadir_range,
                shipgen_kwargs=config.shipgen_kwargs,
                required_surface=expected_surface,
                max_crop_attempts=max(1, surface_crop_attempts),
            )
            if result is None:
                continue
            tile, labels, n_clusters, surface = result
            if surface != expected_surface:
                continue

            name = f"{index:06d}"
            Image.fromarray(tile).save(img_out / f"{name}.png")
            (lbl_out / f"{name}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )
            return len(labels), n_clusters, surface

    fallback_tif = _sample_tif(1 if expected_surface is not None else 0)
    return _compose_for_tif(fallback_tif)


def _write_dataset_yaml(
    output_dir: Path,
    class_id: int,
    *,
    size_thresholds: tuple[float, ...] | None = None,
    params: dict[str, object] | None = None,
) -> None:
    """Write a YOLO dataset YAML config."""
    yaml_path = output_dir / "dataset.yaml"
    lines: list[str] = []

    if params:
        lines.append("# Generation parameters:")
        for key, value in params.items():
            lines.append(f"#   {key}: {value}")
        lines.append("")

    lines.append(f"path: {output_dir.resolve().as_posix()}")
    lines.append("train: images/autosplit_train.txt")
    lines.append("val: images/autosplit_val.txt")
    lines.append("")
    lines.append("names:")
    names = _size_class_names(size_thresholds or ())
    for i, name in enumerate(names):
        lines.append(f"  {class_id + i}: {name}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")