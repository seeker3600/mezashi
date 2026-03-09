import fiftyone as fo
import fiftyone.utils.yolo as fouy

dataset = fo.Dataset.from_dir(
    yaml_path="datasets\\xView_sliced.yaml",
    dataset_type=fo.types.YOLOv5Dataset,
)

session = fo.launch_app(dataset)
session.wait()