"""YOLO OBB dataset loader for FiftyOne.

Converts YOLO OBB labels (9-column: class x1 y1 x2 y2 x3 y3 x4 y4) to
``fo.Polylines`` so that oriented bounding boxes are rendered as proper
quadrilaterals in the FiftyOne App, rather than plain axis-aligned bboxes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import fiftyone as fo

from medetect.yolo.dataset_yaml import (
    choose_splits,
    get_dataset_root,
    load_dataset_yaml,
    resolve_split_dirs,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yolo_obb_dataset(
    yaml_path: str | Path,
    split: str | None = "val",
    *,
    dataset_name: str | None = None,
    overwrite: bool = True,
) -> fo.Dataset:
    """Load a YOLO OBB dataset from a YAML config as a FiftyOne Dataset.

    Labels are converted to ``fo.Polylines`` (4-point closed polygons) so that
    OBBs are visualised correctly in the FiftyOne App.

    Args:
        yaml_path: Path to the YOLO dataset YAML file.
        split: Split to load (``"train"``, ``"val"``, ``"test"``).
            ``None`` loads all available splits.
        dataset_name: FiftyOne dataset name.  Defaults to ``<yaml_stem>_obb``.
        overwrite: Delete an existing FiftyOne dataset with the same name
            before creating a new one.

    Returns:
        A ``fo.Dataset`` with a ``ground_truth`` field of type
        ``fo.Polylines`` and a ``split`` field.
    """
    yaml_path = Path(yaml_path).resolve()

    _, cfg = load_dataset_yaml(yaml_path)
    root = get_dataset_root(cfg, yaml_path, default_to_parent=True)
    class_map = _build_class_map(cfg["names"])

    name = dataset_name or f"{yaml_path.stem}_obb"
    if overwrite and fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = fo.Dataset(name=name)

    splits_to_load = choose_splits(cfg, split)
    samples: list[fo.Sample] = []

    for sp in splits_to_load:
        dirs = resolve_split_dirs(cfg[sp], root)
        for img_path in _iter_images(dirs):
            label_path = _image_to_label_path(img_path)
            sample = fo.Sample(filepath=str(img_path))
            sample["ground_truth"] = _parse_obb_label_file(label_path, class_map)
            sample["split"] = sp
            samples.append(sample)

        logger.info("Loaded %d samples from split=%s", len(samples), sp)

    dataset.add_samples(samples)
    return dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_class_map(names: list | dict) -> dict[int, str]:
    """Convert YAML ``names`` (list or dict) to ``{class_id: label}``."""
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def _iter_images(dirs: list[Path]):
    """Yield image paths from *dirs*, sorted for reproducibility."""
    for d in dirs:
        if not d.is_dir():
            logger.warning("Image directory not found, skipping: %s", d)
            continue
        yield from sorted(p for p in d.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)


def _image_to_label_path(img_path: Path) -> Path:
    """Convert an image path to the corresponding YOLO label ``.txt`` path.

    Assumes the standard YOLO layout ``images/.../foo.jpg`` →
    ``labels/.../foo.txt``.  Falls back to a sibling ``.txt`` file if the
    ``images`` directory segment is not found.
    """
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return img_path.with_suffix(".txt")


def _parse_obb_label_file(label_path: Path, class_map: dict[int, str]) -> fo.Polylines:
    """Parse a YOLO OBB label file into a ``fo.Polylines`` instance.

    Each line must have exactly 9 space-separated values:
    ``class x1 y1 x2 y2 x3 y3 x4 y4`` (normalised coordinates).
    """
    polylines: list[fo.Polyline] = []

    if not label_path.exists():
        return fo.Polylines(polylines=[])

    with label_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            vals = line.split()
            if len(vals) != 9:
                raise ValueError(
                    f"Invalid OBB label at {label_path}:{line_no} "
                    f"(expected 9 columns, got {len(vals)})"
                )

            cls_id = int(vals[0])
            coords = list(map(float, vals[1:]))
            points = [
                [coords[0], coords[1]],
                [coords[2], coords[3]],
                [coords[4], coords[5]],
                [coords[6], coords[7]],
            ]

            polylines.append(
                fo.Polyline(
                    label=class_map.get(cls_id, str(cls_id)),
                    points=[points],
                    closed=True,
                    filled=False,
                )
            )

    return fo.Polylines(polylines=polylines)


def detect_task(yaml_path: Path, split: str = "val") -> Literal["obb", "detect"]:
    """Detect whether a YOLO dataset is OBB or standard detect.

    Samples a few label files from *split* to check the column count.
    Returns ``"obb"`` if 9-column lines are found, else ``"detect"``.
    """
    _, cfg = load_dataset_yaml(yaml_path)
    root = get_dataset_root(cfg, yaml_path, default_to_parent=True)
    splits_to_check = choose_splits(cfg, split) or choose_splits(cfg, None)

    for sp in splits_to_check:
        dirs = resolve_split_dirs(cfg[sp], root)
        for img_dir in dirs:
            label_dir = img_dir.parent.parent / "labels" / img_dir.name
            if not label_dir.is_dir():
                # try sibling labels/ at same level
                label_dir = Path(str(img_dir).replace("images", "labels", 1))
            for txt in sorted(label_dir.glob("*.txt")):
                text = txt.read_text().strip()
                if not text:
                    continue
                n = len(text.split("\n")[0].split())
                if n == 9:
                    return "obb"
                return "detect"

    return "detect"
