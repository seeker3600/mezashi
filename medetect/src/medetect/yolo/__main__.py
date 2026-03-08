from __future__ import annotations

import argparse
import logging
from pathlib import Path

from medetect.yolo.relabel import relabel_yolo_detect_dataset


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
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

    args = parser.parse_args()

    if args.command == "relabel":
        stats = relabel_yolo_detect_dataset(args.config)
        logging.getLogger(__name__).info(
            "relabel complete: files=%d updated=%d relabeled=%d dropped=%d",
            stats["files_processed"],
            stats["files_updated"],
            stats["labels_reassigned"],
            stats["labels_dropped"],
        )


if __name__ == "__main__":
    main()