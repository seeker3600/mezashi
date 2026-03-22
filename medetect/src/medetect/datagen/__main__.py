from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from medetect.datagen.compose import generate_dataset


def _parse_range(s: str) -> tuple[int, int]:
    """Parse ``"min:max"`` into ``(min, max)``."""
    parts = s.split(":")
    if len(parts) != 2:
        msg = f"Expected MIN:MAX, got {s!r}"
        raise argparse.ArgumentTypeError(msg)
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ship detection dataset (YOLO OBB).",
    )
    parser.add_argument(
        "--bg_dir",
        type=Path,
        required=True,
        help="Directory containing Sentinel-2 *_visual.tif background images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for the YOLO dataset.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of training images to generate.",
    )
    parser.add_argument(
        "--ship_dir",
        type=Path,
        default=None,
        help="Directory of pre-generated SVG ship files. "
        "Omit to generate ships on-the-fly.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Output tile size in pixels (default: 640).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Target resolution in m/px. Default: use native GeoTIFF resolution.",
    )
    parser.add_argument(
        "--ships_per_image",
        type=_parse_range,
        default="0:10",
        metavar="MIN:MAX",
        help="Range of ships per image (default: 0:10).",
    )
    parser.add_argument(
        "--cluster_prob",
        type=float,
        default=0.15,
        help="Probability of a cluster instead of a single ship (default: 0.15).",
    )
    parser.add_argument(
        "--cluster_size",
        type=_parse_range,
        default="2:5",
        metavar="MIN:MAX",
        help="Number of ships in a cluster (default: 2:5).",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        help="YOLO class ID for ships (default: 0).",
    )
    parser.add_argument(
        "--erode_coast",
        type=int,
        default=3,
        help="Pixels to erode from coast in water mask (default: 3).",
    )
    parser.add_argument(
        "--min_water_ratio",
        type=float,
        default=0.3,
        help="Minimum water fraction for a usable tile (default: 0.3).",
    )
    parser.add_argument(
        "--ship_blur_sigma",
        type=float,
        default=0.8,
        help="Gaussian blur sigma for ships (default: 0.8).",
    )
    parser.add_argument(
        "--ship_length_min",
        type=float,
        default=None,
        metavar="METRES",
        help="Global minimum ship length in metres. Default: per-class minimum.",
    )
    parser.add_argument(
        "--ship_length_max",
        type=float,
        default=None,
        metavar="METRES",
        help="Global maximum ship length in metres. Default: per-class maximum.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--geo_scale",
        type=float,
        default=None,
        help=(
            "Ignore TIFF geographic CRS and use a fixed pixel scale. "
            "1.0 = 1 input pixel per output pixel, "
            "2.0 = 2 input pixels per output pixel (zoom out), "
            "0.5 = upsample 2x (zoom in). "
            "--resolution still controls ship sizes in metres."
        ),
    )

    args = parser.parse_args()

    ship_length_range: tuple[float, float] | None = None
    if args.ship_length_min is not None or args.ship_length_max is not None:
        lo = args.ship_length_min if args.ship_length_min is not None else 1.0
        hi = args.ship_length_max if args.ship_length_max is not None else 1e9
        ship_length_range = (lo, hi)

    stats = generate_dataset(
        bg_dir=args.bg_dir,
        output_dir=args.output_dir,
        count=args.count,
        ship_dir=args.ship_dir,
        image_size=args.image_size,
        resolution=args.resolution,
        geo_scale=args.geo_scale,
        ships_per_image=args.ships_per_image,
        cluster_prob=args.cluster_prob,
        cluster_size=args.cluster_size,
        class_id=args.class_id,
        erode_coast=args.erode_coast,
        min_water_ratio=args.min_water_ratio,
        ship_blur_sigma=args.ship_blur_sigma,
        ship_length_range=ship_length_range,
        seed=args.seed,
    )

    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
