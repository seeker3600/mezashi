import shutil
from pathlib import Path

import albumentations as A
from ultralytics import YOLO

from medetect.command_history import command_history_path
from medetect.yolo.augment import RandomCloudOverlay
from medetect.yolo.backup import resolve_dataset_root_and_splits

model = "yolo26m-obb.pt"

custom_transforms = [
    A.RandomFog(fog_coef_range=(0.03, 0.10), alpha_coef=0.05, p=0.25),
    RandomCloudOverlay(r"datasets/cloud_overlays_png_640", alpha_range=(0.1, 0.3), p=0.25),
    A.OneOf(
        [
            A.Blur(blur_limit=(3, 15)),
            A.GaussianBlur(blur_limit=(3, 15)),
        ],
        p=1.0,
    ),
    A.GaussNoise(
        std_range=(0.01, 0.04),
        mean_range=(0.0, 0.0),
        p=0.5,
    ),
    A.ImageCompression(
        quality_range=(60, 95),
        compression_type="jpeg",
        p=0.3,
    ),
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
    data="datasets/synthetic_ship_dataset/dataset.yaml",
    imgsz=640,
    # 処理時間関連
    epochs=1,
    batch=2,
    amp=True,
    cache=True,
    workers=8,
    val=True,
    # データ拡張
    degrees=10.0,
    scale=0.1,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.1,
    hsv_s=0.25,
    hsv_v=0.20,
    augmentations=custom_transforms,
    # cutmix=0.0,
    # 無効にするデータ拡張
    mosaic=0.0,
    auto_augment=None,
    erasing=0.0,
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
