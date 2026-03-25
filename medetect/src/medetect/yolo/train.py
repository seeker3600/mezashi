import albumentations as A
from ultralytics import YOLO

from medetect.yolo.augment import RandomCloudOverlay


custom_transforms = [
    A.RandomFog(fog_coef_range=(0.03, 0.10), alpha_coef=0.05, p=0.5),
    RandomCloudOverlay(r"datasets/cloud_overlays_png_640", alpha_range=(0.2, 0.5), p=0.65),
    A.OneOf(
        [
            A.Blur(blur_limit=(3, 30)),
            A.GaussianBlur(blur_limit=(3, 30)),
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
        p=0.25,
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
    epochs=100,
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


def train_yolo_model() -> None:
    YOLO("yolo26m.pt").train(**train_kwargs)