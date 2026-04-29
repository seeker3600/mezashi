from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import fiftyone as fo

from medetect.viewer.obb import detect_task, load_yolo_detect_dataset, load_yolo_obb_dataset


def main() -> None:
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Launch FiftyOne viewer for a YOLO-format dataset YAML."
    )
    parser.add_argument("--yaml", type=Path, required=True, help="Dataset YAML path (YOLO format).")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to view (e.g. 'val', 'train').")
    parser.add_argument(
        "--task",
        choices=["auto", "obb", "detect"],
        default="auto",
        help="Dataset task type.  'auto' detects from label format (default: auto).",
    )

    args = parser.parse_args()
    yaml_path: Path = args.yaml.resolve()

    task = args.task
    if task == "auto":
        task = detect_task(yaml_path, split=args.split)
        logger.info("Auto-detected task: %s", task)

    logger.info("yaml_path: %s", yaml_path)
    logger.info("split: %s", args.split)
    logger.info("task: %s", task)

    if task == "obb":
        dataset = load_yolo_obb_dataset(yaml_path, split=args.split)
    else:
        dataset = load_yolo_detect_dataset(yaml_path, split=args.split)

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
