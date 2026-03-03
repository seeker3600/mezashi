import argparse
import logging
from pathlib import Path
from medetect.xview.convert import convert_xview_to_yolo
from medetect.xview.slice import slice_training_images


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="xView dataset utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # convert サブコマンド
    convert_parser = subparsers.add_parser("convert", help="Convert xView dataset to YOLO format.")
    convert_parser.add_argument("--geojson_path", type=Path, required=True, help="Path to xView geojson file.")
    convert_parser.add_argument("--images_dir", type=Path, required=True, help="Directory containing xView images.")
    convert_parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save YOLO labels.")

    # slice サブコマンド
    slice_parser = subparsers.add_parser("slice", help="Slice training images by resolution.")
    slice_parser.add_argument("--input_dir", type=Path, required=True, help="Input directory (images/train, labels/train).")
    slice_parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for sliced tiles.")
    slice_parser.add_argument("--resolution", type=float, nargs="+", required=True, help="Resolution in m/px. One value for fixed, two for range (min max).")
    slice_parser.add_argument("--image_size", type=int, required=True, help="Output tile size in pixels.")
    slice_parser.add_argument("--overlap", type=float, default=0.0, help="Overlap ratio (0.0-1.0). Default: 0.0.")

    args = parser.parse_args()

    if args.command == "convert":
        convert_xview_to_yolo(
            geojson_path=args.geojson_path,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
        )
    elif args.command == "slice":
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
        )

if __name__ == "__main__":
    main()
