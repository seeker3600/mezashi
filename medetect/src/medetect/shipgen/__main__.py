from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from medetect.shipgen.gen import generate_ships, get_ship_classes


def _parse_types(raw: list[str]) -> dict[str, float]:
    """Parse ``["destroyer:3", "frigate:2"]`` into ``{name: weight}``."""
    result: dict[str, float] = {}
    for token in raw:
        if ":" not in token:
            msg = f"Invalid type spec {token!r}; expected NAME:WEIGHT"
            raise argparse.ArgumentTypeError(msg)
        name, weight_str = token.rsplit(":", 1)
        result[name] = float(weight_str)
    return result


def _parse_float_range(s: str) -> tuple[float, float]:
    """Parse ``"min:max"`` into ``(float, float)``."""
    parts = s.split(":")
    if len(parts) != 2:
        msg = f"Expected MIN:MAX, got {s!r}"
        raise argparse.ArgumentTypeError(msg)
    return float(parts[0]), float(parts[1])


def main() -> None:
    public_classes = get_ship_classes()
    debug_classes = sorted(set(get_ship_classes(include_debug=True)) - set(public_classes))
    available_classes = public_classes + [f"{name} (debug only)" for name in debug_classes]

    parser = argparse.ArgumentParser(
        description="Generate synthetic ship silhouette SVGs for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for generated SVG files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of SVG files to generate.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        metavar="CLASS:WEIGHT",
        help=(
            "Ship classes with sampling weights (e.g. destroyer:3 frigate:2). "
            f"Available classes: {', '.join(available_classes)}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--hull_noise",
        type=float,
        default=0.005,
        help="Hull outline perturbation scale (default: 0.005).",
    )
    parser.add_argument(
        "--n_hull_points",
        type=int,
        default=64,
        help="Polygon sample points per hull side (default: 64).",
    )
    parser.add_argument(
        "--deck_scatter_density",
        type=float,
        default=3.0,
        help=(
            "Deck scatter density: shapes per unit L/B ratio. "
            "0 disables scatter entirely (default: 3.0)."
        ),
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Remove and recreate --output_dir if it already exists.",
    )
    parser.add_argument(
        "--filetype",
        choices=["svg", "png"],
        default="svg",
        help="Output file format: svg (default) or png (for debugging).",
    )
    parser.add_argument(
        "--offnadir",
        type=_parse_float_range,
        default=(0.0, 0.0),
        metavar="MIN:MAX",
        help=(
            "Off-nadir viewing angle range in degrees (default: 0:0 = nadir only). "
            "Each ship independently draws offnadir_deg ~ Uniform(MIN, MAX) "
            "and sensor_az_ship_deg ~ Uniform(0, 360) unless --sensor-az-deg is set."
        ),
    )
    parser.add_argument(
        "--sensor-az-deg",
        type=float,
        default=None,
        help=(
            "Optional fixed sensor azimuth in ship frame. 0 = bow-on, 90 = starboard, "
            "180 = stern-on, 270 = port. When omitted, each ship samples its own azimuth."
        ),
    )

    args = parser.parse_args()

    if args.output_dir.exists():
        if args.override:
            shutil.rmtree(args.output_dir)
        else:
            parser.error(f"--output_dir already exists: {args.output_dir}. Use --override to overwrite.")

    types: dict[str, float] | None = None
    if args.types is not None:
        types = _parse_types(args.types)

    generate_ships(
        output_dir=args.output_dir,
        count=args.count,
        types=types,
        seed=args.seed,
        hull_noise=args.hull_noise,
        n_hull_points=args.n_hull_points,
        deck_scatter_density=args.deck_scatter_density,
        filetype=args.filetype,
        offnadir_range=args.offnadir,
        sensor_az_ship_deg=args.sensor_az_deg,
    )


if __name__ == "__main__":
    main()
