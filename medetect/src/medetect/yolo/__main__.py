from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from medetect.command_history import append_command_history
from medetect.yolo.backup import resolve_dataset_root_and_splits


def convert_tiffs_to_png(*args, **kwargs):
    from medetect.yolo.tiff2png import convert_tiffs_to_png as _convert_tiffs_to_png

    return _convert_tiffs_to_png(*args, **kwargs)


def relabel_yolo_detect_dataset(*args, **kwargs):
    from medetect.yolo.relabel import relabel_yolo_detect_dataset as _relabel_yolo_detect_dataset

    return _relabel_yolo_detect_dataset(*args, **kwargs)


def train_yolo_model(*args, **kwargs):
    from medetect.yolo.train import train_yolo_model as _train_yolo_model

    return _train_yolo_model(*args, **kwargs)


def split_train_to_val(*args, **kwargs):
    from medetect.yolo.valsplit import split_train_to_val as _split_train_to_val

    return _split_train_to_val(*args, **kwargs)


def restore_dataset_splits(*args, **kwargs):
    from medetect.yolo.backup import restore_dataset_splits as _restore_dataset_splits

    return _restore_dataset_splits(*args, **kwargs)


def expand_obb_dataset(*args, **kwargs):
    from medetect.yolo.expand_obb import expand_obb_dataset as _expand_obb_dataset

    return _expand_obb_dataset(*args, **kwargs)


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

    valsplit_parser = subparsers.add_parser(
        "valsplit",
        help="Extract a fraction of train images, augment them, and save as val.",
    )
    valsplit_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset YAML path.",
    )
    valsplit_parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of train images to move to val (0.0-1.0 exclusive).",
    )
    valsplit_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Target image size (default: 640).",
    )
    valsplit_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible selection.",
    )
    valsplit_parser.add_argument(
        "--disable-augs",
        nargs="*",
        default=[],
        metavar="KEY",
        help="Augmentation keys to disable (e.g. degrees translate scale shear perspective).",
    )

    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore one or more dataset splits from _backup.",
    )
    restore_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset YAML path.",
    )
    restore_parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        metavar="SPLIT",
        help="Split names to restore (default: all splits defined in the dataset YAML).",
    )
    restore_parser.add_argument(
        "--with-images",
        action="store_true",
        help="Restore images as well as labels.",
    )

    subparsers.add_parser("train", help="Train a YOLO model.")

    expand_parser = subparsers.add_parser(
        "expand-obb",
        help="Expand OBB width/height in a YOLO OBB dataset.",
    )
    expand_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset YAML path.",
    )
    expand_parser.add_argument(
        "--expand-height",
        type=float,
        default=0.0,
        metavar="N",
        help="Expand OBB height (longer dimension) by N pixels (constant).",
    )
    expand_parser.add_argument(
        "--expand-width",
        type=float,
        default=0.0,
        metavar="N",
        help="Expand OBB width (shorter dimension) by N pixels (constant).",
    )
    expand_parser.add_argument(
        "--expand-height-weighted",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "Expand OBB height with inverse-proportional weighting. "
            "Actual expansion = N × (median_height / current_height)."
        ),
    )
    expand_parser.add_argument(
        "--expand-width-weighted",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "Expand OBB width with inverse-proportional weighting. "
            "Actual expansion = N × (median_width / current_width)."
        ),
    )
    expand_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of worker threads (default: CPU count; 0 disables parallelism for profiling).",
    )
    expand_parser.add_argument(
        "--avoid-overlap",
        action="store_true",
        help=(
            "Scale each requested expansion back so the expanded OBB does not "
            "overlap any other original OBB in the same image."
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
        stats = relabel_yolo_detect_dataset(
            args.config,
            empty_image_ratio=args.empty_image_ratio,
            max_workers=args.workers,
        )
        append_command_history(
            resolve_dataset_root_and_splits(args.config)[0],
            command="yolo relabel",
            result=stats,
        )
    elif args.command == "train":
        train_yolo_model()
    elif args.command == "valsplit":
        split_train_to_val(
            config=args.config,
            fraction=args.fraction,
            imgsz=args.imgsz,
            seed=args.seed,
            disable_augs=args.disable_augs,
        )
        append_command_history(
            resolve_dataset_root_and_splits(args.config)[0],
            command="yolo valsplit",
        )
    elif args.command == "restore":
        restore_dataset_splits(
            args.config,
            splits=args.splits,
            with_images=args.with_images,
        )
        append_command_history(
            resolve_dataset_root_and_splits(args.config)[0],
            command="yolo restore",
        )
    elif args.command == "expand-obb":
        stats = expand_obb_dataset(
            args.config,
            expand_height=args.expand_height,
            expand_width=args.expand_width,
            expand_height_weighted=args.expand_height_weighted,
            expand_width_weighted=args.expand_width_weighted,
            avoid_overlap=args.avoid_overlap,
            max_workers=args.workers,
        )
        append_command_history(
            resolve_dataset_root_and_splits(args.config)[0],
            command="yolo expand-obb",
            result=stats,
        )

if __name__ == "__main__":
    main()