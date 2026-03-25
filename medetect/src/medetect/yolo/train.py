import albumentations as A
from ultralytics import YOLO


custom_transforms = [
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.20, p=0.3),
    A.Blur(blur_limit=3, p=0.1),
]


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
    hsv_s=0.4,
    hsv_v=0.3,
    augmentations=custom_transforms,
    # 無効にするデータ拡張
    mosaic=0.0,
    erasing=0.0,
    auto_augment=None,
)


def train_yolo_model() -> None:
    YOLO("yolo26m.pt").train(**train_kwargs)