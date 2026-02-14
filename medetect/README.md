# medetect

## とりあえずモデルを作る

```ps
pixi install
yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=50 imgsz=512 batch=0.75 amp=True
yolo export model=yolo26n-obb.pt format=onnx opset=20
```

## とりあえず推論する

```ps
yolo obb predict `
    model=runs/obb/train/weights/best.pt `
    source=src.png `
    imgsz=512 `
    conf=0.25 `
    save=True
```
