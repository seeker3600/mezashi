from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from medetect.yolo.relabel import relabel_yolo_detect_dataset
from medetect.yolo.train import train_yolo_model


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO dataset utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    relabel_parser = subparsers.add_parser(
        "relabel",
        help="Rewrite YOLO detect labels according to merges in a dataset YAML.",
    )
    relabel_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset YAML path containing path and merges.",
    )
    relabel_parser.add_argument(
        "--empty-image-keep-prob",
        type=float,
        default=None,
        help=(
            "Target ratio of empty-label images to total images after relabeling (0.0-1.0). "
            "Images are randomly removed until the ratio reaches this value. "
            "0.0 removes all empty-label images; 1.0 keeps all; 0.5 keeps at most as many "
            "empty-label images as labeled ones. Has no effect if the ratio is already below the target."
        ),
    )

    train_parser = subparsers.add_parser("train", help="Train a YOLO model.")
    train_parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help=(
            "Maximum number of times to retry training after a CUDA out-of-memory error. "
            "Each retry resumes from the latest checkpoint. Default: 0 (no retry)."
        ),
    )

    args = parser.parse_args()

    if args.command == "relabel":
        relabel_yolo_detect_dataset(
            args.config,
            empty_image_keep_prob=args.empty_image_keep_prob,
        )
    elif args.command == "train":
        train_yolo_model(max_retries=args.max_retries)

if __name__ == "__main__":
    main()