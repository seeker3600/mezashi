"""Create an augmented validation split from training data."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from pathlib import Path

import re

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths
from ultralytics.utils.ops import xywhr2xyxyxyxy

from medetect.yolo.train import train_kwargs

logger = logging.getLogger(__name__)

# Keys in train_kwargs that configure data augmentation.
_AUG_KEYS = (
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "auto_augment",
    "erasing",
    "augmentations",
)

# Value that disables each augmentation key.
_AUG_DISABLED_VALUES: dict[str, object] = {
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "auto_augment": None,
    "erasing": 0.0,
    "augmentations": None,
}


def _update_yaml_paths(config: Path, updates: dict[str, str]) -> None:
    """Update specific keys in a YAML file without destroying comments."""
    text = config.read_text(encoding="utf-8")
    for key, value in updates.items():
        # Match the value token (not starting with '#') and replace it;
        # any trailing inline comment is left untouched.
        text = re.sub(
            rf'^({re.escape(key)}\s*:)[ \t]*[^#\s]\S*',
            rf'\1 {value}',
            text,
            flags=re.MULTILINE,
        )
    config.write_text(text, encoding="utf-8")


def _build_hyp(disable_augs: Sequence[str] = ()):
    """Build an ultralytics hyp namespace from *train_kwargs* augmentation settings.

    Parameters
    ----------
    disable_augs:
        Augmentation key names to force-disable.  Each key is set to its
        neutral value (0.0 or ``None``) regardless of *train_kwargs*.
    """
    unknown = set(disable_augs) - set(_AUG_DISABLED_VALUES)
    if unknown:
        raise ValueError(
            f"Unknown augmentation keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(_AUG_DISABLED_VALUES)}"
        )
    hyp = get_cfg()
    for key in _AUG_KEYS:
        if key in train_kwargs:
            setattr(hyp, key, train_kwargs[key])
    for key in disable_augs:
        setattr(hyp, key, _AUG_DISABLED_VALUES[key])
    return hyp


def _detect_task(label_dir: Path) -> str:
    """Return ``'obb'`` if labels use 8-coord OBB format, else ``'detect'``."""
    for txt in sorted(label_dir.glob("*.txt")):
        text = txt.read_text().strip()
        if not text:
            continue
        n_values = len(text.split("\n")[0].split())
        if n_values == 9:  # class + 4 xy-pairs
            return "obb"
        return "detect"
    return "detect"


def _save_yolo_labels(path: Path, cls: np.ndarray, bboxes: np.ndarray) -> None:
    """Write YOLO labels to *path* (supports both detect and OBB formats)."""
    lines: list[str] = []
    for c, b in zip(cls, bboxes):
        c_val = int(c[0]) if c.ndim > 0 else int(c)
        coords = " ".join(f"{v:.6f}" for v in b)
        lines.append(f"{c_val} {coords}")
    path.write_text("\n".join(lines))


def split_train_to_val(
    config: Path,
    fraction: float,
    imgsz: int = 640,
    seed: int | None = None,
    disable_augs: Sequence[str] = (),
) -> None:
    """Extract a fraction of train images, augment them, and write to val.

    On the first call the selected train images are moved to *val_before* as a
    reversible backup, and augmented copies are written to *val*.  On
    subsequent calls *val_before* is used as the source so the operation is
    idempotent and reversible: original data is always preserved in
    *val_before*.

    Uses ``YOLODataset.build_transforms(hyp)`` (which delegates to
    ``v8_transforms``) to apply the same augmentation pipeline configured
    in *train.py*.  Bounding-box labels are updated by the geometric
    transforms automatically.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    with open(config) as f:
        data: dict = yaml.safe_load(f)

    dataset_path = Path(data["path"]).resolve()
    val_before_images = dataset_path / "images" / "val_before"
    val_before_labels = dataset_path / "labels" / "val_before"
    val_images_dir = dataset_path / "images" / "val"
    val_labels_dir = dataset_path / "labels" / "val"

    first_run = not (
        val_before_images.exists() and any(val_before_images.iterdir())
    )

    hyp = _build_hyp(disable_augs=disable_augs)

    if first_run:
        train_path = dataset_path / data["train"]

        # If the YAML references an autosplit .txt that doesn't exist yet,
        # fall back to the directory that would contain the txt file (images/).
        if train_path.suffix == ".txt" and not train_path.exists():
            train_path = train_path.parent
            logger.warning(
                "train txt not found; using directory: %s", train_path
            )

        train_labels_dir = dataset_path / "labels" / "train"
        task = _detect_task(train_labels_dir)
        is_obb = task == "obb"

        # YOLODataset.__init__ calls build_transforms(hyp), which uses
        # v8_transforms(dataset, imgsz, hyp) when augment=True.
        dataset = YOLODataset(
            img_path=str(train_path),
            data=data,
            task=task,
            augment=True,
            hyp=hyp,
            imgsz=imgsz,
        )

        n = len(dataset)
        n_val = max(1, round(n * fraction))

        if seed is not None:
            random.seed(seed)
        indices = sorted(random.sample(range(n), n_val))

        val_before_images.mkdir(parents=True, exist_ok=True)
        val_before_labels.mkdir(parents=True, exist_ok=True)
        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        orig_im_files = [Path(dataset.labels[i]["im_file"]) for i in indices]
        orig_lbl_files = [
            Path(p) for p in img2label_paths([str(f) for f in orig_im_files])
        ]

        for idx, orig_im, orig_lbl in tqdm(
            zip(indices, orig_im_files, orig_lbl_files),
            total=n_val,
            desc="Augmenting val",
        ):
            sample = dataset[idx]
            img_t = sample["img"]  # (C, H, W) uint8, RGB
            cls_np = sample["cls"].numpy()  # (n_obj, 1)
            bboxes_raw = sample["bboxes"].numpy()
            if is_obb:
                # xywhr (N, 5) → xyxyxyxy (N, 4, 2) → (N, 8)
                bboxes_np = xywhr2xyxyxyxy(sample["bboxes"]).numpy().reshape(-1, 8)
            else:
                bboxes_np = bboxes_raw  # (n_obj, 4) normalised xywh

            # CHW RGB → HWC BGR for cv2.imwrite
            img_bgr = np.ascontiguousarray(
                img_t.numpy().transpose(1, 2, 0)[:, :, ::-1]
            )

            stem = orig_im.stem
            cv2.imwrite(str(val_images_dir / f"{stem}.png"), img_bgr)
            _save_yolo_labels(val_labels_dir / f"{stem}.txt", cls_np, bboxes_np)

            # Move originals to val_before (preserve for idempotent re-runs)
            orig_im.rename(val_before_images / orig_im.name)
            if orig_lbl.exists():
                orig_lbl.rename(val_before_labels / orig_lbl.name)

        logger.info(
            "Moved %d / %d train images to val_before (augmented to val)",
            n_val,
            n,
        )

        # Update dataset.yaml to use plain directories (preserves comments)
        _update_yaml_paths(config, {"train": "images/train", "val": "images/val"})
        logger.info("Updated %s: train/val now point to directories", config.name)

        # Remove autosplit_*.txt files
        for txt in sorted((dataset_path / "images").glob("autosplit_*.txt")):
            txt.unlink()
            logger.info("Removed %s", txt.name)

    else:
        # Subsequent run: val_before exists — regenerate val from it.
        task = _detect_task(val_before_labels)
        is_obb = task == "obb"

        dataset = YOLODataset(
            img_path=str(val_before_images),
            data=data,
            task=task,
            augment=True,
            hyp=hyp,
            imgsz=imgsz,
        )

        indices = list(range(len(dataset)))
        orig_im_files = [Path(dataset.labels[i]["im_file"]) for i in indices]

        val_images_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        for idx, orig_im in tqdm(
            zip(indices, orig_im_files),
            total=len(indices),
            desc="Regenerating val",
        ):
            sample = dataset[idx]
            img_t = sample["img"]  # (C, H, W) uint8, RGB
            cls_np = sample["cls"].numpy()  # (n_obj, 1)
            bboxes_raw = sample["bboxes"].numpy()
            if is_obb:
                bboxes_np = xywhr2xyxyxyxy(sample["bboxes"]).numpy().reshape(-1, 8)
            else:
                bboxes_np = bboxes_raw

            # CHW RGB → HWC BGR for cv2.imwrite
            img_bgr = np.ascontiguousarray(
                img_t.numpy().transpose(1, 2, 0)[:, :, ::-1]
            )

            stem = orig_im.stem
            cv2.imwrite(str(val_images_dir / f"{stem}.png"), img_bgr)
            _save_yolo_labels(val_labels_dir / f"{stem}.txt", cls_np, bboxes_np)

        logger.info("Regenerated %d val images from val_before", len(indices))
