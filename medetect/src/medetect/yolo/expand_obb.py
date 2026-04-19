"""Expand OBB (Oriented Bounding Box) dimensions in a YOLO dataset.

height = longer dimension of OBB, width = shorter dimension.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import logging
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_OVERLAP_AREA_EPSILON = 1e-9


@dataclass
class _ObbRecord:
    line_index: int
    class_id: int
    corners: np.ndarray
    raw_line: str


def _obb_dimensions(corners: np.ndarray) -> tuple[float, float]:
    """Return (width, height) of an OBB in pixel space.

    Width is the shorter dimension, height is the longer.
    *corners*: shape (4, 2), pixel coordinates going around the box.
    """
    edge_a = float(np.linalg.norm(corners[1] - corners[0]))
    edge_b = float(np.linalg.norm(corners[2] - corners[1]))
    return (min(edge_a, edge_b), max(edge_a, edge_b))


def _expand_obb(
    corners: np.ndarray,
    expand_width: float = 0.0,
    expand_height: float = 0.0,
) -> np.ndarray:
    """Expand OBB corners by the given pixel amounts.

    *expand_width*: pixels to add to the shorter dimension.
    *expand_height*: pixels to add to the longer dimension.
    *corners*: shape (4, 2) pixel coordinates in order p0-p1-p2-p3.

    The box is parameterised as ``center ± u ± v`` where
    ``u = (p1-p0)/2`` and ``v = (p2-p1)/2``.
    """
    if expand_width == 0 and expand_height == 0:
        return corners.copy()

    center = corners.mean(axis=0)
    u = (corners[1] - corners[0]) / 2.0
    v = (corners[2] - corners[1]) / 2.0

    len_u = float(np.linalg.norm(u)) * 2.0
    len_v = float(np.linalg.norm(v)) * 2.0

    if len_u <= 0 or len_v <= 0:
        return corners.copy()

    # Map expand_width → shorter edge, expand_height → longer edge.
    if len_u <= len_v:
        delta_u, delta_v = expand_width, expand_height
    else:
        delta_u, delta_v = expand_height, expand_width

    if delta_u != 0:
        u_hat = u / np.linalg.norm(u)
        u = u + u_hat * delta_u / 2.0
    if delta_v != 0:
        v_hat = v / np.linalg.norm(v)
        v = v + v_hat * delta_v / 2.0

    return np.array([
        center - u - v,
        center + u - v,
        center + u + v,
        center - u + v,
    ])


def _clip_corners_to_image(corners: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Clamp pixel-space corners to the image boundary."""
    clipped = corners.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(img_w))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(img_h))
    return clipped


def _format_obb_line(
    class_id: int,
    corners: np.ndarray,
    img_w: int,
    img_h: int,
) -> str:
    """Serialize pixel-space OBB corners as one YOLO OBB label line."""
    corners_norm = corners.copy()
    corners_norm[:, 0] = np.clip(corners_norm[:, 0] / img_w, 0.0, 1.0)
    corners_norm[:, 1] = np.clip(corners_norm[:, 1] / img_h, 0.0, 1.0)
    coord_str = " ".join(f"{value:.6f}" for value in corners_norm.flatten())
    return f"{class_id} {coord_str}"


def _compute_total_expansion(
    corners: np.ndarray,
    *,
    expand_width: float,
    expand_height: float,
    expand_width_weighted: float,
    expand_height_weighted: float,
    median_width: float,
    median_height: float,
) -> tuple[float, float]:
    """Return total width/height expansion for one OBB in pixel space."""
    width, height = _obb_dimensions(corners)

    total_w = expand_width
    total_h = expand_height
    if expand_width_weighted > 0 and width > 0 and median_width > 0:
        total_w += expand_width_weighted * (median_width / width)
    if expand_height_weighted > 0 and height > 0 and median_height > 0:
        total_h += expand_height_weighted * (median_height / height)
    return total_w, total_h


def _obb_polygon(corners: np.ndarray):
    """Build a polygon for overlap checks."""
    return Polygon(corners).buffer(0)


def _has_overlap(candidate, obstacles: list) -> bool:
    """Return True when a candidate polygon overlaps any obstacle polygon by area."""
    for obstacle in obstacles:
        if float(candidate.intersection(obstacle).area) > _OVERLAP_AREA_EPSILON:
            return True
    return False


def _parse_label_text(
    text: str,
    img_w: int,
    img_h: int,
) -> tuple[list[str | None], list[_ObbRecord]]:
    """Split a label file into OBB records and preserved raw lines."""
    output_lines: list[str | None] = []
    records: list[_ObbRecord] = []

    for line_index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            output_lines.append("")
            continue

        tokens = stripped.split()
        if len(tokens) != 9:
            output_lines.append(stripped)
            continue

        class_id = int(tokens[0])
        coords = [float(token) for token in tokens[1:]]
        corners = np.array(coords, dtype=float).reshape(4, 2)
        corners[:, 0] *= img_w
        corners[:, 1] *= img_h
        records.append(
            _ObbRecord(
                line_index=line_index,
                class_id=class_id,
                corners=corners,
                raw_line=stripped,
            )
        )
        output_lines.append(None)

    return output_lines, records


def _expand_without_overlap(
    records: list[_ObbRecord],
    img_w: int,
    img_h: int,
    *,
    expand_width: float,
    expand_height: float,
    expand_width_weighted: float,
    expand_height_weighted: float,
    median_width: float,
    median_height: float,
) -> tuple[list[np.ndarray], int]:
    """Expand OBBs greedily while preserving non-overlap within one image."""
    if not records:
        return [], 0

    target_expansions = [
        _compute_total_expansion(
            record.corners,
            expand_width=expand_width,
            expand_height=expand_height,
            expand_width_weighted=expand_width_weighted,
            expand_height_weighted=expand_height_weighted,
            median_width=median_width,
            median_height=median_height,
        )
        for record in records
    ]

    original_corners = [_clip_corners_to_image(record.corners, img_w, img_h) for record in records]
    final_corners = [corners.copy() for corners in original_corners]
    final_polygons = [_obb_polygon(corners) for corners in original_corners]
    initially_overlapping: set[int] = set()

    for index, polygon in enumerate(final_polygons):
        for other_index in range(index + 1, len(final_polygons)):
            if float(polygon.intersection(final_polygons[other_index]).area) > _OVERLAP_AREA_EPSILON:
                initially_overlapping.add(index)
                initially_overlapping.add(other_index)

    priorities: list[tuple[float, int, int]] = []
    for index, (record, (total_w, total_h)) in enumerate(zip(records, target_expansions, strict=False)):
        if total_w == 0.0 and total_h == 0.0:
            continue
        width, height = _obb_dimensions(record.corners)
        additional_area = max((width + total_w) * (height + total_h) - width * height, 0.0)
        priorities.append((-additional_area, record.line_index, index))

    labels_expanded = 0
    for _, _, index in sorted(priorities):
        if index in initially_overlapping:
            continue

        total_w, total_h = target_expansions[index]
        record = records[index]
        obstacles = [
            final_polygons[other_index]
            for other_index in range(len(records))
            if other_index != index
        ]

        best_corners = original_corners[index]
        full_corners = _clip_corners_to_image(
            _expand_obb(record.corners, expand_width=total_w, expand_height=total_h),
            img_w,
            img_h,
        )
        full_polygon = _obb_polygon(full_corners)
        if not _has_overlap(full_polygon, obstacles):
            best_corners = full_corners
        else:
            low = 0.0
            high = 1.0
            for _ in range(24):
                mid = (low + high) / 2.0
                candidate_corners = _clip_corners_to_image(
                    _expand_obb(
                        record.corners,
                        expand_width=total_w * mid,
                        expand_height=total_h * mid,
                    ),
                    img_w,
                    img_h,
                )
                candidate_polygon = _obb_polygon(candidate_corners)
                if _has_overlap(candidate_polygon, obstacles):
                    high = mid
                else:
                    low = mid
                    best_corners = candidate_corners

        final_corners[index] = best_corners
        final_polygons[index] = _obb_polygon(best_corners)
        if not np.allclose(best_corners, original_corners[index], atol=1e-6):
            labels_expanded += 1

    return final_corners, labels_expanded


def _get_image_size(
    label_path: Path,
    dataset_root: Path,
) -> tuple[int, int] | None:
    """Return ``(width, height)`` of the image corresponding to *label_path*."""
    try:
        relative_label = label_path.relative_to(dataset_root / "labels")
    except ValueError:
        return None

    image_dir = dataset_root / "images" / relative_label.parent
    stem = label_path.stem

    for ext in _IMAGE_EXTENSIONS:
        image_path = image_dir / f"{stem}{ext}"
        if image_path.exists():
            with Image.open(image_path) as img:
                return img.size  # (width, height)
    return None


def _process_label_file(
    label_path: Path,
    img_w: int,
    img_h: int,
    *,
    expand_width: float,
    expand_height: float,
    expand_width_weighted: float,
    expand_height_weighted: float,
    median_width: float,
    median_height: float,
    avoid_overlap: bool = False,
) -> dict[str, int]:
    """Expand OBBs in a single label file. Returns update stats."""
    text = label_path.read_text(encoding="utf-8")
    output_lines, records = _parse_label_text(text, img_w, img_h)

    if avoid_overlap:
        final_corners, labels_expanded = _expand_without_overlap(
            records,
            img_w,
            img_h,
            expand_width=expand_width,
            expand_height=expand_height,
            expand_width_weighted=expand_width_weighted,
            expand_height_weighted=expand_height_weighted,
            median_width=median_width,
            median_height=median_height,
        )
        for record, corners in zip(records, final_corners, strict=False):
            if np.allclose(corners, _clip_corners_to_image(record.corners, img_w, img_h), atol=1e-6):
                output_lines[record.line_index] = record.raw_line
            else:
                output_lines[record.line_index] = _format_obb_line(
                    record.class_id,
                    corners,
                    img_w,
                    img_h,
                )
    else:
        labels_expanded = 0
        for record in records:
            total_w, total_h = _compute_total_expansion(
                record.corners,
                expand_width=expand_width,
                expand_height=expand_height,
                expand_width_weighted=expand_width_weighted,
                expand_height_weighted=expand_height_weighted,
                median_width=median_width,
                median_height=median_height,
            )

            if total_w != 0 or total_h != 0:
                corners = _clip_corners_to_image(
                    _expand_obb(record.corners, expand_width=total_w, expand_height=total_h),
                    img_w,
                    img_h,
                )
                labels_expanded += 1
                output_lines[record.line_index] = _format_obb_line(
                    record.class_id,
                    corners,
                    img_w,
                    img_h,
                )
            else:
                output_lines[record.line_index] = record.raw_line

    new_text = "\n".join(line if line is not None else "" for line in output_lines)
    if output_lines:
        new_text += "\n"

    updated = int(new_text != text)
    if updated:
        label_path.write_text(new_text, encoding="utf-8")

    return {"updated": updated, "labels_expanded": labels_expanded}


# ------------------------------------------------------------------
# Two-pass helpers for weighted expansion
# ------------------------------------------------------------------

def _collect_one(
    label_path: Path,
    dataset_root: Path,
) -> tuple[Path, tuple[int, int] | None, list[float], list[float]]:
    """Read one label file and return its image size and OBB dimensions."""
    size = _get_image_size(label_path, dataset_root)
    if size is None:
        return label_path, None, [], []
    img_w, img_h = size
    text = label_path.read_text(encoding="utf-8")
    widths: list[float] = []
    heights: list[float] = []
    for line in text.strip().splitlines():
        tokens = line.strip().split()
        if len(tokens) != 9:
            continue
        coords = [float(t) for t in tokens[1:]]
        corners = np.array(coords).reshape(4, 2)
        corners[:, 0] *= img_w
        corners[:, 1] *= img_h
        w, h = _obb_dimensions(corners)
        widths.append(w)
        heights.append(h)
    return label_path, size, widths, heights


_BACKUP_DIR_NAME = "labels_before_expand"


def _backup_labels(dataset_root: Path) -> None:
    """Copy ``labels/`` to ``labels_before_expand/`` (first run only)."""
    backup = dataset_root / _BACKUP_DIR_NAME
    labels = dataset_root / "labels"
    shutil.copytree(labels, backup)
    logger.info("Backed up labels to %s", backup)


def _restore_labels_from_backup(dataset_root: Path) -> None:
    """Overwrite ``labels/`` with contents from ``labels_before_expand/``."""
    backup = dataset_root / _BACKUP_DIR_NAME
    labels = dataset_root / "labels"
    shutil.rmtree(labels)
    shutil.copytree(backup, labels)
    logger.info("Restored labels from %s", backup)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def expand_obb_dataset(
    dataset_root: Path,
    *,
    expand_height: float = 0.0,
    expand_width: float = 0.0,
    expand_height_weighted: float = 0.0,
    expand_width_weighted: float = 0.0,
    avoid_overlap: bool = False,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Expand OBB dimensions in a YOLO OBB dataset.

    Parameters
    ----------
    dataset_root:
        Root of the YOLO dataset (must contain ``labels/``).
    expand_height:
        Constant pixel expansion for the longer OBB dimension.
    expand_width:
        Constant pixel expansion for the shorter OBB dimension.
    expand_height_weighted:
        Base pixel expansion for height with inverse-proportional weighting.
        Actual expansion = *base* × (median_height / current_height).
    expand_width_weighted:
        Base pixel expansion for width with inverse-proportional weighting.
        Actual expansion = *base* × (median_width / current_width).
    avoid_overlap:
        If True, greedily scale expansions back within each image so OBBs do not
        acquire new overlap area with one another.
    max_workers:
        Thread pool size.  Defaults to CPU count.
    """
    dataset_root = Path(dataset_root).resolve()
    labels_root = dataset_root / "labels"

    if not labels_root.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_root}")

    backup_root = dataset_root / _BACKUP_DIR_NAME
    if backup_root.exists() and any(backup_root.iterdir()):
        # Subsequent run — restore original labels before re-expanding.
        _restore_labels_from_backup(dataset_root)
    else:
        # First run — create backup of untouched labels.
        _backup_labels(dataset_root)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    label_paths = sorted(labels_root.rglob("*.txt"))
    if not label_paths:
        logger.warning("No label files found in %s", labels_root)
        return {"files_processed": 0, "files_updated": 0, "labels_expanded": 0}

    need_weighted = expand_height_weighted > 0 or expand_width_weighted > 0
    img_sizes: dict[Path, tuple[int, int]] = {}
    median_width = 0.0
    median_height = 0.0

    if need_weighted:
        # Pass 1: collect OBB dimensions to compute medians.
        all_widths: list[float] = []
        all_heights: list[float] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_collect_one, lp, dataset_root): lp
                for lp in label_paths
            }
            with tqdm(total=len(futures), desc="collecting dimensions", unit="file", dynamic_ncols=True) as pbar:
                for f in concurrent.futures.as_completed(futures):
                    lp, size, ws, hs = f.result()
                    if size is not None:
                        img_sizes[lp] = size
                        all_widths.extend(ws)
                        all_heights.extend(hs)
                    pbar.update(1)

        median_width = float(np.median(all_widths)) if all_widths else 0.0
        median_height = float(np.median(all_heights)) if all_heights else 0.0
        logger.info(
            "OBB medians: width=%.1f px, height=%.1f px (%d boxes)",
            median_width, median_height, len(all_widths),
        )
    else:
        # Single pass — just gather image sizes.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_get_image_size, lp, dataset_root): lp
                for lp in label_paths
            }
            with tqdm(total=len(futures), desc="reading image sizes", unit="file", dynamic_ncols=True) as pbar:
                for f in concurrent.futures.as_completed(futures):
                    lp = futures[f]
                    size = f.result()
                    if size is not None:
                        img_sizes[lp] = size
                    pbar.update(1)

    # Pass 2 (or only pass): expand OBBs.
    stats = {"files_processed": 0, "files_updated": 0, "labels_expanded": 0}

    def _process(lp: Path) -> dict[str, int]:
        size = img_sizes.get(lp)
        if size is None:
            return {"updated": 0, "labels_expanded": 0}
        return _process_label_file(
            lp,
            size[0],
            size[1],
            expand_width=expand_width,
            expand_height=expand_height,
            expand_width_weighted=expand_width_weighted,
            expand_height_weighted=expand_height_weighted,
            median_width=median_width,
            median_height=median_height,
            avoid_overlap=avoid_overlap,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, lp): lp for lp in label_paths}
        with tqdm(total=len(futures), desc="expanding OBBs", unit="file", dynamic_ncols=True) as pbar:
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                stats["files_processed"] += 1
                stats["files_updated"] += result["updated"]
                stats["labels_expanded"] += result["labels_expanded"]
                pbar.update(1)

    logger.info(
        "expand-obb complete: files=%d updated=%d labels=%d",
        stats["files_processed"],
        stats["files_updated"],
        stats["labels_expanded"],
    )
    return stats
