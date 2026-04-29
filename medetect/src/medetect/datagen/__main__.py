from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logging.getLogger("rasterio").setLevel(logging.WARNING)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)


from medetect.command_history import append_command_history
from medetect.datagen import generate_dataset


def _parse_range(s: str) -> tuple[int, int]:
    """Parse ``"min:max"`` into ``(min, max)``."""
    parts = s.split(":")
    if len(parts) != 2:
        msg = f"Expected MIN:MAX, got {s!r}"
        raise argparse.ArgumentTypeError(msg)
    return int(parts[0]), int(parts[1])


def _parse_float_range(s: str) -> tuple[float, float]:
    """Parse ``"min:max"`` into ``(float, float)``."""
    parts = s.split(":")
    if len(parts) != 2:
        msg = f"Expected MIN:MAX, got {s!r}"
        raise argparse.ArgumentTypeError(msg)
    return float(parts[0]), float(parts[1])


def _parse_hex_color(s: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` into an RGB tuple."""
    if len(s) != 7 or not s.startswith("#"):
        msg = f"Expected #RRGGBB, got {s!r}"
        raise argparse.ArgumentTypeError(msg)

    try:
        return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
    except ValueError as exc:
        msg = f"Expected #RRGGBB, got {s!r}"
        raise argparse.ArgumentTypeError(msg) from exc


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
        help=(
            "Range of placement events per image (single ship or cluster group, "
            "default: 0:10)."
        ),
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
        "--cluster_mixed_prob",
        type=float,
        default=0.5,
        help=(
            "Probability that a cluster contains mixed ship types and sizes "
            "rather than uniform sister ships (default: 0.5). "
            "0.0 = always uniform (same size), 1.0 = always mixed."
        ),
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
        "--ship_length",
        type=_parse_float_range,
        default=None,
        metavar="MIN:MAX",
        help="Global ship length range in metres (e.g. 5:150). Default: per-class range.",
    )
    parser.add_argument(
        "--length_exponent",
        type=float,
        default=1.0,
        help=(
            "Controls the ship size-frequency distribution. "
            "1.0 = log-uniform (default, naturally more small ships). "
            "> 1.0 = even more small ships. "
            "< 1.0 toward 0 = more uniform (less small-biased)."
        ),
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
    parser.add_argument(
        "--ship_alpha",
        type=_parse_float_range,
        default="0.7:0.95",
        metavar="MIN:MAX",
        help="Ship opacity range for blending (default: 0.7:0.95).",
    )
    parser.add_argument(
        "--size_threshold",
        type=float,
        default=None,
        help=(
            "Ship length threshold in metres for two-class labelling. "
            "Ships shorter than this → ship_small, at or above → ship_large. "
            "Omit for single 'ship' class (default)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker processes "
            "(default: os.cpu_count(); 0 disables parallelism for profiling)."
        ),
    )
    parser.add_argument(
        "--wake_prob_scale",
        type=float,
        default=1.0,
        help=(
            "single ships only: wake occurrence probability multiplier "
            "(default: 1.0). "
            "Applied to the built-in per-state probabilities "
            "(STOPPED 20%%, SLOW 50%%, MEDIUM 80%%, FAST 90%%). "
            "Clustered ships currently do not render wakes. "
            "0.0 = never generate wakes. "
            "0.5 = halve all probabilities. "
            "2.0 = double all probabilities (capped at 100%% per state)."
        ),
    )
    parser.add_argument(
        "--wake_alpha_scale",
        type=float,
        default=1.0,
        help=(
            "single ships only: wake opacity/intensity multiplier "
            "(default: 1.0). "
            "Controls how strongly a wake appears when it is generated. "
            "Clustered ships currently do not render wakes. "
            "0.0 = fully transparent (disables rendering). "
            "1.5 = 50%% brighter wakes. "
            "Independent of --wake_prob_scale: probability and intensity "
            "can be tuned separately."
        ),
    )
    parser.add_argument(
        "--debug_bg_color",
        type=_parse_hex_color,
        default=None,
        metavar="#RRGGBB",
        help=(
            "Force synthetic backgrounds to a flat debug colour after tile selection "
            "(debug only)."
        ),
    )
    parser.add_argument(
        "--shadow_alpha_scale",
        type=float,
        default=5.0,
        help=(
            "Shadow intensity multiplier for ship water shadows (default: 5.0). "
            "Applied to both single ships and clustered ships. "
            "0.0 disables shadow rendering. Values above 1.0 make shadows darker."
        ),
    )
    parser.add_argument(
        "--shadow_length",
        type=_parse_float_range,
        default="0.0:1.5",
        metavar="MIN:MAX",
        help=(
            "Normalized cast-shadow length multiplier range (default: 0.0:1.5). "
            "The sampled value is uniform across the range and is applied relative "
            "to the estimated ship height. 0.0 means no cast shadow."
        ),
    )
    parser.add_argument(
        "--false_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing false-negative source images (PNG/TIFF). "
            "When set, tiles are cropped from these images and added to the "
            "dataset with empty labels. Requires --false_ratio."
        ),
    )
    parser.add_argument(
        "--false_ratio",
        type=float,
        default=0.0,
        metavar="RATIO",
        help=(
            "Fraction of the total dataset (--count) that should be false-negative "
            "images (0.0–<1.0, default: 0.0). "
            "E.g. --count 100 --false_ratio 0.2 generates 80 synthetic + 20 false "
            "negatives = 100 total. Requires --false_dir."
        ),
    )
    parser.add_argument(
        "--coastline",
        type=Path,
        default=None,
        metavar="SHP",
        help=(
            "Path to an OSM coastline shapefile (lines.shp) in EPSG:4326. "
            "When set, coastline geometries provide precise land/water "
            "boundaries to prevent ships from being placed on land."
        ),
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override the output directory if it already exists.",
    )
    args = parser.parse_args()

    if args.false_ratio and args.false_dir is None:
        parser.error("--false_ratio requires --false_dir")
    if args.false_dir and not (0.0 < args.false_ratio < 1.0):
        parser.error("--false_ratio must be in (0, 1) when --false_dir is set")

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
        cluster_mixed_prob=args.cluster_mixed_prob,
        class_id=args.class_id,
        erode_coast=args.erode_coast,
        min_water_ratio=args.min_water_ratio,
        ship_blur_sigma=args.ship_blur_sigma,
        ship_alpha=args.ship_alpha,
        ship_length_range=args.ship_length,
        length_exponent=args.length_exponent,
        seed=args.seed,
        size_threshold=args.size_threshold,
        wake_prob_scale=args.wake_prob_scale,
        wake_alpha_scale=args.wake_alpha_scale,
        debug_bg_color=args.debug_bg_color,
        shadow_alpha_scale=args.shadow_alpha_scale,
        shadow_length_range=args.shadow_length,
        false_dir=args.false_dir,
        false_ratio=args.false_ratio,
        max_workers=args.workers,
        coastline=args.coastline,
        override=args.override,
    )
    append_command_history(
        args.output_dir,
        command="datagen",
        result=stats,
    )

    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
