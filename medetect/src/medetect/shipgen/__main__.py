from __future__ import annotations

import argparse
import logging
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


def main() -> None:
    all_classes = get_ship_classes()

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
            f"Available classes: {', '.join(all_classes)}"
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

    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
