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
from medetect.yolo.tiff2png import convert_tiffs_to_png
from medetect.yolo.train import train_yolo_model


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO dataset utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tiff2png_parser = subparsers.add_parser(
        "tiff2png",
        help="Recursively convert TIFF/GeoTIFF files to PNG under a directory.",
    )
    tiff2png_parser.add_argument(
        "dir",
        type=Path,
        help="Root directory to search for TIFF files.",
    )
    tiff2png_parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the original TIFF files after conversion (default: delete them).",
    )
    tiff2png_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of worker threads (default: CPU count).",
    )

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
        "--empty-image-ratio",
        type=float,
        default=None,
        help=(
            "Target ratio of empty-label images to total images after relabeling (0.0-1.0). "
            "Images are randomly removed until the ratio reaches this value. "
            "0.0 removes all empty-label images; 1.0 keeps all; 0.5 keeps at most as many "
            "empty-label images as labeled ones. Has no effect if the ratio is already below the target."
        ),
    )
    relabel_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of worker threads (default: CPU count).",
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

    if args.command == "tiff2png":
        convert_tiffs_to_png(
            args.dir,
            delete_source=not args.keep_source,
            max_workers=args.workers,
        )
    elif args.command == "relabel":
        relabel_yolo_detect_dataset(
            args.config,
            empty_image_ratio=args.empty_image_ratio,
            max_workers=args.workers,
        )
    elif args.command == "train":
        train_yolo_model(max_retries=args.max_retries)

if __name__ == "__main__":
    main()