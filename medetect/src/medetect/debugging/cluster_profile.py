"""Cluster profile helpers for YOLO OBB datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from medetect.debugging.pixel_profile import (
    extract_line_profile,
    save_profile_visualization,
    write_profile_table,
)


def parse_yolo_obb_label(text: str) -> list[tuple[int, list[float]]]:
    """Parse YOLO OBB rows into ``(class_id, coords)`` records."""
    results = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) != 9:
            continue
        results.append((int(parts[0]), [float(value) for value in parts[1:]]))
    return results


def obb_center(coords: list[float]) -> tuple[float, float]:
    """Return the center of an OBB in normalized coordinates."""
    xs = coords[0::2]
    ys = coords[1::2]
    return sum(xs) / 4, sum(ys) / 4


def obb_midpoints(coords: list[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the two short-axis midpoints of an OBB."""
    points = [(coords[index * 2], coords[index * 2 + 1]) for index in range(4)]
    m01 = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
    m23 = ((points[2][0] + points[3][0]) / 2, (points[2][1] + points[3][1]) / 2)
    return m01, m23


def find_best_cluster(label_dir: Path) -> tuple[Path | None, list[float] | None, int]:
    """Return the densest cluster found under a label directory."""
    best_file: Path | None = None
    best_cluster: list[float] | None = None
    best_count = 0

    for txt_path in sorted(label_dir.glob("*.txt")):
        rows = parse_yolo_obb_label(txt_path.read_text(encoding="utf-8"))
        clusters = [coords for class_id, coords in rows if class_id == 2]
        ships = [coords for class_id, coords in rows if class_id == 1]
        for cluster in clusters:
            cx_min = min(cluster[0::2])
            cx_max = max(cluster[0::2])
            cy_min = min(cluster[1::2])
            cy_max = max(cluster[1::2])
            count = sum(
                1
                for ship in ships
                if cx_min <= obb_center(ship)[0] <= cx_max and cy_min <= obb_center(ship)[1] <= cy_max
            )
            if count > best_count:
                best_count = count
                best_cluster = cluster
                best_file = txt_path

    return best_file, best_cluster, best_count


def cluster_cross_line(
    cluster_obb: list[float],
    width: int,
    height: int,
    *,
    extend: float = 0.1,
) -> tuple[int, int, int, int]:
    """Return a line crossing the cluster short axis in pixel coordinates."""
    m01, m23 = obb_midpoints(cluster_obb)
    dx = m01[0] - m23[0]
    dy = m01[1] - m23[1]
    length = max(np.hypot(dx, dy), 1e-9)
    nx, ny = dx / length, dy / length
    cx_avg = (m01[0] + m23[0]) / 2
    cy_avg = (m01[1] + m23[1]) / 2
    half = length / 2 + extend
    sx = (cx_avg - nx * half) * width
    sy = (cy_avg - ny * half) * height
    ex = (cx_avg + nx * half) * width
    ey = (cy_avg + ny * half) * height
    return int(round(sx)), int(round(sy)), int(round(ex)), int(round(ey))


def run_cluster_profile(
    dataset_dir: Path,
    *,
    split: str = "train",
    output_path: Path = Path("debug_runs/cluster-profile/cluster_profile.png"),
) -> Path:
    """Generate a cluster-crossing profile image and print TSV data to stdout."""
    label_dir = dataset_dir / "labels" / split
    image_dir = dataset_dir / "images" / split

    print(f"Searching for densest cluster in {label_dir} ...", file=sys.stderr)
    label_file, cluster_obb, ship_count = find_best_cluster(label_dir)
    if label_file is None or cluster_obb is None:
        raise ValueError(f"No clusters found under {label_dir}")

    stem = label_file.stem
    image_path = None
    for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            image_path = candidate
            break
    if image_path is None:
        raise FileNotFoundError(f"Image not found for {stem}")

    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image)
    x0, y0, x1, y1 = cluster_cross_line(cluster_obb, *image.size)
    positions, values = extract_line_profile(rgb, x0, y0, x1, y1)

    print(f"Best cluster: {label_file.name}  ships={ship_count}", file=sys.stderr)
    print(f"Line: ({x0}, {y0}) -> ({x1}, {y1})  [image {image.size[0]}x{image.size[1]}]", file=sys.stderr)
    write_profile_table(positions, values, sys.stdout, print_all=False)
    save_profile_visualization(rgb, positions, values, (x0, y0), (x1, y1), output_path)
    print(f"\nSaved -> {output_path}", file=sys.stderr)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-cluster pixel profile")
    parser.add_argument("dataset_dir", type=Path, help="Dataset root containing images/ and labels/")
    parser.add_argument("--split", default="train", help="Split name (default: train)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_runs/cluster-profile/cluster_profile.png"),
        help="Output image path.",
    )
    args = parser.parse_args(argv)
    run_cluster_profile(args.dataset_dir, split=args.split, output_path=args.output)
