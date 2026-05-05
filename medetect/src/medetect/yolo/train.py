import shutil
from pathlib import Path

import albumentations as A
from ultralytics import YOLO

from medetect.command_history import command_history_path
from medetect.yolo.augment import RandomCloudOverlay
from medetect.yolo.backup import resolve_dataset_root_and_splits

model = "yolo26n-obb.pt"

custom_transforms = [
    # RandomCloudOverlay(r"/mnt/r/training/datasets/cloud_overlays_png_640", alpha_range=(0.1, 0.3), p=0.3),
    A.GaussNoise(std_range=(0.01, 0.1), mean_range=(0.0, 0.0), p=0.8),
    A.GaussianBlur(blur_limit=(3,3), sigma_range=(0.1, 0.5), p=0.3),
    A.ChromaticAberration(
        primary_distortion_limit=(0.005, 0.02),
        secondary_distortion_limit=(0.005, 0.02),
        p=0.05,
    ),
    A.ISONoise(
        color_shift=(0.002, 0.005),
        intensity=(0.05, 0.12),
        p=0.2,
    ),
    #A.ImageCompression(
    #    quality_range=(85, 95),
    #    compression_type="jpeg",
    #    p=0.2,
    #),
    A.RandomBrightnessContrast(
        brightness_limit=0.12,
        contrast_limit=0.18,
        p=0.1,
    ),
    A.RandomGamma(
        gamma_limit=(90, 110),
        p=0.1,
    ),
    A.CLAHE(
        clip_limit=(1.0, 2.0),
        tile_grid_size=(8, 8),
        p=0.1,
    ),
]


train_kwargs: dict = dict(
    # 基本設定
    data="/root/training/datasets/synthetic_ship_dataset/dataset.yaml",
    imgsz=1280,
    # 処理時間関連
    epochs=50,
    # patience=30,
    batch=2,
    amp=False,
    cache='disk',
    workers=8,
    val=True,
    # データ拡張
    degrees=20.0,
    translate=0.1,
    scale=0.0,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.1,
    # hsv_s=0.25,
    # hsv_v=0.20,
    augmentations=custom_transforms,
    # cutmix=0.0,
    # 無効にするデータ拡張
    mosaic=0.0,
    auto_augment=None,
    erasing=0.0,
    # conf=0.05,
    iou=0.7,
    max_det=1000,
    agnostic_nms=False,
    multi_scale=0.0,
    rect=True,
    deterministic=True,
    #device='cpu',
)


def _copy_training_artifacts(save_dir: str | Path, dataset: str | Path) -> None:
    run_dir = Path(save_dir)
    dataset_root, _ = resolve_dataset_root_and_splits(dataset)
    dataset_history = command_history_path(dataset_root)
    if dataset_history.is_file():
        shutil.copy(dataset_history, run_dir / dataset_history.name)
    shutil.copy(Path(__file__), run_dir / "train.py")


def train_yolo_model() -> None:
    results = YOLO(model).train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        return
    _copy_training_artifacts(save_dir, train_kwargs["data"])
