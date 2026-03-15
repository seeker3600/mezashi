import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from medetect.xview.slice import slice_training_images


def main():
    parser = argparse.ArgumentParser(description="xView dataset utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # slice サブコマンド
    slice_parser = subparsers.add_parser("slice", help="Slice training images by resolution.")
    slice_parser.add_argument("--input_dir", type=Path, required=True, help="Input directory (images/train, labels/train).")
    slice_parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for sliced tiles.")
    slice_parser.add_argument("--resolution", type=float, nargs="+", required=True, help="Resolution in m/px. One value for fixed, two for range (min max).")
    slice_parser.add_argument("--image_size", type=int, required=True, help="Output tile size in pixels.")
    slice_parser.add_argument("--overlap", type=float, default=0.0, help="Overlap ratio (0.0-1.0). Default: 0.0.")
    slice_parser.add_argument("--min_area_ratio", type=float, default=0.1, help="Min area ratio for bbox clipping (0.0-1.0). 0.0: include all, 1.0: fully inside only. Default: 0.1.")
    slice_parser.add_argument("--max_images", type=int, default=None, help="Max number of images to process (debug).")
    slice_parser.add_argument("--output_geotiff", action="store_true", help="Output GeoTIFF files for sliced images.")

    args = parser.parse_args()

    if args.command == "slice":
        resolution_vals = args.resolution
        if len(resolution_vals) == 1:
            resolution = resolution_vals[0]
        elif len(resolution_vals) == 2:
            resolution = (resolution_vals[0], resolution_vals[1])
        else:
            parser.error("--resolution requires 1 or 2 values.")

        slice_training_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            resolution=resolution,
            image_size=args.image_size,
            overlap=args.overlap,
            min_area_ratio=args.min_area_ratio,
            max_images=args.max_images,
            output_geotiff=args.output_geotiff,
        )

if __name__ == "__main__":
    main()
