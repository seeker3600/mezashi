from ultralytics import YOLO
import albumentations as A
import os

def train_yolo_model():

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

    model.train(
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