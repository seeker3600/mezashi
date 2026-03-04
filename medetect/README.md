# medetect

## dota8

```bash
yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=50 imgsz=512 batch=0.75 amp=True

yolo export model=yolo26n-obb-dota8.pt format=onnx opset=20 imgsz=512
```

## xView

```bash
export OPENCV_LOG_LEVEL="ERROR"
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_ALLOC_CONF

# workers=0 device=cpu 
yolo detect train data=xView.yaml model=yolo26n.pt imgsz=640 \
  epochs=100 batch=0.3 amp=True cache=disk workers=2 val=False \
  mosaic=1.0 scale=0.6 translate=0.2 degrees=10 fliplr=0.5 flipud=0.0

yolo export model=yolo26n-detect-xview.pt format=onnx opset=20 imgsz=640
```

## とりあえず推論する

```bash
yolo obb predict \
    model=best.pt \
    source=src.png \
    imgsz=512 \
    conf=0.25 \
    save=True

yolo detect predict \
    model=best.pt \
    source=src.png \
    imgsz=640 \
    conf=0.05 \
    save=True
```
