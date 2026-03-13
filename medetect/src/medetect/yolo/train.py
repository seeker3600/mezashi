from __future__ import annotations

import logging
import os
from pathlib import Path

import albumentations as A
import torch
from ultralytics import YOLO

_logger = logging.getLogger(__name__)


def _find_latest_checkpoint() -> Path | None:
    """Return the most recently modified ``last.pt`` under ``runs/detect/``."""
    base = Path("runs/detect")
    if not base.is_dir():
        return None
    checkpoints = sorted(
        base.glob("train*/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def _is_oom_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* represents a CUDA out-of-memory error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def train_yolo_model(max_retries: int = 0) -> None:

    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ['PYTORCH_ALLOC_CONF']
    

    model = YOLO("yolo26m.pt")

    custom_transforms = [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.20,
            p=0.3,
        ),
        # A.RandomGamma(gamma_limit=(85, 115), p=0.2),
        A.Blur(blur_limit=3, p=0.1),
    ]

    # yolo detect train data=xView_sliced.yaml model=yolo26m.pt imgsz=640 \
    #     epochs=150 batch=0.3 amp=True cache=disk workers=2 val=True \
    #     scale=0.5 fliplr=0.3 flipud=0.3 \
    #     mosaic=0.0 erasing=0.0 auto_augment=None

    train_kwargs: dict = dict(
        # 基本設定
        data="xView_sliced.yaml",
        imgsz=640,
        # 処理時間関連
        epochs=150,
        batch=0.3,
        amp=True,
        cache="disk",
        workers=2,
        val=True,
        # データ拡張
        scale=0.5,
        fliplr=0.3,
        flipud=0.3,
        # hsv_h=0.0,
        hsv_s=0.4,
        hsv_v=0.3,
        augmentations=custom_transforms,
        # cutmix=0.0,
        # 無効にするデータ拡張
        mosaic=0.0,
        erasing=0.0,
        auto_augment=None,
    )

    for attempt in range(max_retries + 1):
        try:
            model.train(**train_kwargs)
            return
        except BaseException as exc:
            if not _is_oom_error(exc) or attempt >= max_retries:
                raise
            last_pt = _find_latest_checkpoint()
            if last_pt is None:
                _logger.warning(
                    "OOM on attempt %d/%d but no checkpoint found; giving up.",
                    attempt + 1,
                    max_retries + 1,
                )
                raise
            _logger.warning(
                "OOM on attempt %d/%d; retrying from checkpoint %s.",
                attempt + 1,
                max_retries + 1,
                last_pt,
            )
            model = YOLO(str(last_pt))
            train_kwargs = {"resume": True}