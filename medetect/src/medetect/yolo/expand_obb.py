"""Expand OBB (Oriented Bounding Box) dimensions in a YOLO dataset.

height = longer dimension of OBB, width = shorter dimension.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


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
) -> dict[str, int]:
    """Expand OBBs in a single label file. Returns update stats."""
    text = label_path.read_text(encoding="utf-8")
    new_lines: list[str] = []
    labels_expanded = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) != 9:
            # Not an OBB line — preserve as-is.
            new_lines.append(stripped)
            continue

        class_id = int(tokens[0])
        coords = [float(t) for t in tokens[1:]]
        corners = np.array(coords).reshape(4, 2)
        corners[:, 0] *= img_w
        corners[:, 1] *= img_h

        w, h = _obb_dimensions(corners)

        total_w = expand_width
        total_h = expand_height
        if expand_width_weighted > 0 and w > 0 and median_width > 0:
            total_w += expand_width_weighted * (median_width / w)
        if expand_height_weighted > 0 and h > 0 and median_height > 0:
            total_h += expand_height_weighted * (median_height / h)

        if total_w != 0 or total_h != 0:
            corners = _expand_obb(corners, expand_width=total_w, expand_height=total_h)
            labels_expanded += 1
            # Normalise back and clamp.
            corners[:, 0] = np.clip(corners[:, 0] / img_w, 0.0, 1.0)
            corners[:, 1] = np.clip(corners[:, 1] / img_h, 0.0, 1.0)
            coord_str = " ".join(f"{v:.6f}" for v in corners.flatten())
            new_lines.append(f"{class_id} {coord_str}")
        else:
            new_lines.append(stripped)

    new_text = "\n".join(new_lines)
    if new_lines:
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
    max_workers:
        Thread pool size.  Defaults to CPU count.
    """
    dataset_root = Path(dataset_root).resolve()
    labels_root = dataset_root / "labels"

    if not labels_root.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_root}")

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
