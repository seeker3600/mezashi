"""Create an augmented validation split from training data."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths

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


def _build_hyp():
    """Build an ultralytics hyp namespace from *train_kwargs* augmentation settings."""
    hyp = get_cfg()
    for key in _AUG_KEYS:
        if key in train_kwargs:
            setattr(hyp, key, train_kwargs[key])
    return hyp


def _save_yolo_labels(path: Path, cls: np.ndarray, bboxes: np.ndarray) -> None:
    """Write YOLO detection labels (``cls cx cy w h``) to *path*."""
    lines: list[str] = []
    for c, b in zip(cls, bboxes):
        c_val = int(c[0]) if c.ndim > 0 else int(c)
        lines.append(f"{c_val} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
    path.write_text("\n".join(lines))


def split_train_to_val(
    config: Path,
    fraction: float,
    imgsz: int = 640,
    seed: int | None = None,
) -> None:
    """Extract a fraction of train images, augment them, and write to val.

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
    train_path = dataset_path / data["train"]

    hyp = _build_hyp()

    # YOLODataset.__init__ calls build_transforms(hyp), which uses
    # v8_transforms(dataset, imgsz, hyp) when augment=True.
    dataset = YOLODataset(
        img_path=str(train_path),
        data=data,
        augment=True,
        hyp=hyp,
        imgsz=imgsz,
    )

    n = len(dataset)
    n_val = max(1, round(n * fraction))

    if seed is not None:
        random.seed(seed)
    indices = sorted(random.sample(range(n), n_val))

    val_images_dir = dataset_path / "images" / "val"
    val_labels_dir = dataset_path / "labels" / "val"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute original label paths
    orig_im_files = [Path(dataset.labels[i]["im_file"]) for i in indices]
    orig_lbl_files = [
        Path(p) for p in img2label_paths([str(f) for f in orig_im_files])
    ]

    for idx, orig_im, orig_lbl in zip(indices, orig_im_files, orig_lbl_files):
        sample = dataset[idx]
        img_t = sample["img"]  # (C, H, W) uint8, RGB
        cls_np = sample["cls"].numpy()  # (n_obj, 1)
        bboxes_np = sample["bboxes"].numpy()  # (n_obj, 4) normalised xywh

        # CHW RGB → HWC BGR for cv2.imwrite
        img_bgr = np.ascontiguousarray(
            img_t.numpy().transpose(1, 2, 0)[:, :, ::-1]
        )

        stem = orig_im.stem
        cv2.imwrite(str(val_images_dir / f"{stem}.png"), img_bgr)
        _save_yolo_labels(val_labels_dir / f"{stem}.txt", cls_np, bboxes_np)

        # Remove originals
        orig_im.unlink(missing_ok=True)
        orig_lbl.unlink(missing_ok=True)

        logger.info("Processed %s", stem)

    logger.info("Moved %d / %d train images to val (augmented)", n_val, n)

    # Update dataset.yaml to use plain directories
    data["train"] = "images/train"
    data["val"] = "images/val"
    with open(config, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    logger.info("Updated %s: train/val now point to directories", config.name)

    # Remove autosplit_*.txt files
    for txt in sorted((dataset_path / "images").glob("autosplit_*.txt")):
        txt.unlink()
        logger.info("Removed %s", txt.name)
