from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from ultralytics.utils import DATASETS_DIR

import fiftyone as fo

def main() -> None:
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Launch FiftyOne viewer for a YOLO-format dataset YAML."
    )
    parser.add_argument("yaml", type=Path, help="Dataset YAML path (YOLO format).")
    args = parser.parse_args()

    yaml_path: Path = args.yaml.resolve()

    logger.info("dataset_dir: %s", DATASETS_DIR)
    logger.info("yaml_path: %s", yaml_path)

    yaml_dst = Path(DATASETS_DIR) / "dataset.yaml"
    shutil.copy(yaml_path, yaml_dst)

    dataset = fo.Dataset.from_dir(
        yaml_path=str(yaml_dst),
        dataset_type=fo.types.YOLOv5Dataset,
    )

    session = fo.launch_app(dataset)
    session.wait()

    os.remove(yaml_dst)

if __name__ == "__main__":
    main()
