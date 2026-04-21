"""Pixel profile helpers shared by debugging commands and tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
from PIL import Image, ImageDraw


def extract_line_profile(
    arr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel values sampled along the line from ``(x0, y0)`` to ``(x1, y1)``."""
    n = max(int(np.hypot(x1 - x0, y1 - y0)), 1)
    xs = np.round(np.linspace(x0, x1, n + 1)).astype(int)
    ys = np.round(np.linspace(y0, y1, n + 1)).astype(int)

    height, width = arr.shape[:2]
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    positions = np.linspace(0.0, float(np.hypot(x1 - x0, y1 - y0)), n + 1)
    values = arr[ys, xs]
    return positions, values


def normalize_values(values: np.ndarray) -> np.ndarray:
    """Normalize pixel values from [0, 255] to [0.0, 1.0]."""
    return values.astype(np.float32) / 255.0


def resolve_coords(
    args_coords: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert coordinates to absolute pixel indices.

    Values are treated as normalized when all four lie in [0, 1].
    Otherwise they are interpreted as absolute pixel indices.
    """
    x1, y1, x2, y2 = args_coords
    if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        px0 = int(round(x1 * (width - 1)))
        py0 = int(round(y1 * (height - 1)))
        px1 = int(round(x2 * (width - 1)))
        py1 = int(round(y2 * (height - 1)))
    else:
        px0, py0 = int(round(x1)), int(round(y1))
        px1, py1 = int(round(x2)), int(round(y2))
    return px0, py0, px1, py1


def sample_image_profile(
    image_path: Path,
    coords: tuple[float, float, float, float],
    *,
    grayscale: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Load an image and sample a line profile from it."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = resolve_coords(coords, width, height)
    array = np.array(image.convert("L")) if grayscale else np.array(image)
    positions, values = extract_line_profile(array, x0, y0, x1, y1)
    return positions, values, (x0, y0), (x1, y1), (width, height)


def format_profile_table(
    positions: np.ndarray,
    values: np.ndarray,
    *,
    normalize: bool = False,
    print_all: bool = False,
) -> list[str]:
    """Format sampled profile values as TSV rows."""
    output_values = normalize_values(values) if normalize else values

    if output_values.ndim == 1:
        header = "pos_px\tvalue"

        def row(position: float, value: np.ndarray) -> str:
            if normalize:
                rendered = format(float(value), ".4f")
            else:
                rendered = str(int(value))
            return f"{position:.1f}\t{rendered}"
    else:
        channel_count = output_values.shape[1]
        names = ["R", "G", "B"][:channel_count]
        header = "pos_px\t" + "\t".join(names)

        def row(position: float, value: np.ndarray) -> str:
            if normalize:
                rendered = "\t".join(format(float(component), ".4f") for component in value)
            else:
                rendered = "\t".join(str(int(component)) for component in value)
            return f"{position:.1f}\t{rendered}"

    rows = [header]
    count = len(positions)
    if print_all or count <= 20:
        rows.extend(row(position, value) for position, value in zip(positions, output_values, strict=True))
        return rows

    rows.extend(row(position, value) for position, value in zip(positions[:5], output_values[:5], strict=True))
    rows.append(f"... ({count - 10} rows omitted) ...")
    rows.extend(row(position, value) for position, value in zip(positions[-5:], output_values[-5:], strict=True))
    return rows


def write_profile_table(
    positions: np.ndarray,
    values: np.ndarray,
    output: TextIO,
    *,
    normalize: bool = False,
    print_all: bool = False,
) -> None:
    """Write a TSV representation of a profile to a text stream."""
    output.write("\n".join(format_profile_table(positions, values, normalize=normalize, print_all=print_all)))
    output.write("\n")


def save_profile_visualization(
    image_rgb: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    output_path: Path,
) -> None:
    """Save a composite image showing the sampled line and its profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, y0 = start
    x1, y1 = end
    vis = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(vis)
    draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 0), width=2)
    draw.ellipse([(x0 - 5, y0 - 5), (x0 + 5, y0 + 5)], fill=(255, 50, 50))
    draw.ellipse([(x1 - 5, y1 - 5), (x1 + 5, y1 + 5)], fill=(50, 200, 50))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(np.array(vis))
    axes[0].set_title("Image  (start/end marked)")
    axes[0].axis("off")

    plot_ax = axes[1]
    if values.ndim == 1:
        plot_ax.plot(positions, values, color="dimgray", linewidth=1.5)
        plot_ax.set_ylabel("Intensity (0-255)")
    else:
        palette = ["#e53935", "#43a047", "#1e88e5"]
        for channel, (color, label) in enumerate(zip(palette, ["R", "G", "B"], strict=True)):
            plot_ax.plot(
                positions,
                values[:, channel],
                color=color,
                label=label,
                linewidth=1.0,
                alpha=0.85,
            )
        plot_ax.legend()
        plot_ax.set_ylabel("Value (0-255)")

    plot_ax.set_xlabel("Distance along line (pixels)")
    plot_ax.set_title("Pixel profile")
    plot_ax.set_ylim(0, 270)
    plot_ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract pixel values along a line segment in an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("x1", type=float, help="Start X (0-1 or pixel index)")
    parser.add_argument("y1", type=float, help="Start Y (0-1 or pixel index)")
    parser.add_argument("x2", type=float, help="End X (0-1 or pixel index)")
    parser.add_argument("y2", type=float, help="End Y (0-1 or pixel index)")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale before sampling")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_runs/pixel-profile/profile_output.png"),
        help="Output composite image path.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print every sampled pixel to stdout instead of a compact summary.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Output pixel values as 0.0-1.0 instead of 0-255.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = resolve_coords((args.x1, args.y1, args.x2, args.y2), width, height)

    print(
        f"Line: ({x0}, {y0}) -> ({x1}, {y1})  [image {width}x{height}]",
        file=sys.stderr,
    )
    array = np.array(image.convert("L")) if args.grayscale else np.array(image)
    positions, values = extract_line_profile(array, x0, y0, x1, y1)
    write_profile_table(
        positions,
        values,
        sys.stdout,
        normalize=args.normalize,
        print_all=args.print_all,
    )
    save_profile_visualization(np.array(image), positions, values, (x0, y0), (x1, y1), args.output)
    print(f"\nSaved composite image -> {args.output}", file=sys.stderr)
