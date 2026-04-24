from __future__ import annotations

import argparse
import sys
from pathlib import Path

DESCRIPTION = "Debugging and visual QA utilities for medetect."

from medetect.debugging import DEFAULT_SHIPGEN_QA_CLASSES, run_shipgen_profile_qa
from medetect.debugging.cluster_profile import main as cluster_profile_main
from medetect.debugging.pixel_profile import main as pixel_profile_main
from medetect.debugging.shadow_preview import render_shadow_previews
from medetect.debugging.ship_preview import DEFAULT_PREVIEW_CLASSES, save_ship_previews
from medetect.debugging.wake_preview import render_wake_previews


def _parse_hex_color(value: str) -> tuple[int, int, int]:
    if len(value) != 7 or not value.startswith("#"):
        raise argparse.ArgumentTypeError(f"Expected #RRGGBB, got {value!r}")
    try:
        return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected #RRGGBB, got {value!r}") from exc


def _normalize_variant_choice(value: str) -> str | None:
    return None if value == "auto" else value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pixel_parser = subparsers.add_parser("pixel-profile", help="Extract a pixel profile from an image.")
    pixel_parser.add_argument("image")
    pixel_parser.add_argument("x1")
    pixel_parser.add_argument("y1")
    pixel_parser.add_argument("x2")
    pixel_parser.add_argument("y2")
    pixel_parser.add_argument("extra", nargs=argparse.REMAINDER)

    cluster_parser = subparsers.add_parser("cluster-profile", help="Profile the densest ship cluster in a dataset.")
    cluster_parser.add_argument("dataset_dir", type=Path)
    cluster_parser.add_argument("--split", default="train")
    cluster_parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_runs/cluster-profile/cluster_profile.png"),
    )

    preview_parser = subparsers.add_parser("ship-preview", help="Render ship preview grids.")
    preview_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_runs/ship-preview"),
    )
    preview_parser.add_argument("--classes", nargs="+", default=list(DEFAULT_PREVIEW_CLASSES))
    preview_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123, 999])
    preview_parser.add_argument("--bg-color", type=_parse_hex_color, default=(40, 60, 90))
    preview_parser.add_argument(
        "--trim-mode",
        choices=["auto", "none", "perimeter", "bow"],
        default="auto",
    )
    preview_parser.add_argument(
        "--visible-side",
        choices=["auto", "none", "port", "starboard"],
        default="auto",
    )

    shipgen_qa_parser = subparsers.add_parser("shipgen-qa", help="Run the standard 10-ship beam-profile QA.")
    shipgen_qa_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_runs/shipgen-profile-qa"),
    )
    shipgen_qa_parser.add_argument("--classes", nargs="+", default=list(DEFAULT_SHIPGEN_QA_CLASSES))
    shipgen_qa_parser.add_argument("--seed", type=int, default=42)
    shipgen_qa_parser.add_argument("--beam-px", type=int, default=128)
    shipgen_qa_parser.add_argument("--length-px", type=int, default=640)
    shipgen_qa_parser.add_argument("--bg-color", type=_parse_hex_color, default=(40, 60, 90))
    shipgen_qa_parser.add_argument(
        "--trim-mode",
        choices=["auto", "none", "perimeter", "bow"],
        default="none",
    )
    shipgen_qa_parser.add_argument(
        "--visible-side",
        choices=["auto", "none", "port", "starboard"],
        default="none",
    )

    shadow_parser = subparsers.add_parser("shadow-preview", help="Render shadow QA preview images.")
    shadow_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_runs/shadow-preview"),
    )

    wake_parser = subparsers.add_parser("wake-preview", help="Render wake QA preview images.")
    wake_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_runs/wake-preview"),
    )

    args = parser.parse_args(argv)

    if args.command == "pixel-profile":
        pixel_profile_main([args.image, args.x1, args.y1, args.x2, args.y2, *args.extra])
        return
    if args.command == "cluster-profile":
        cluster_profile_main([
            str(args.dataset_dir),
            "--split",
            args.split,
            "--output",
            str(args.output),
        ])
        return
    if args.command == "ship-preview":
        outputs = save_ship_previews(
            args.output_dir,
            classes=args.classes,
            seeds=args.seeds,
            bg_color=args.bg_color,
            trim_mode=_normalize_variant_choice(args.trim_mode),
            visible_side=_normalize_variant_choice(args.visible_side),
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    if args.command == "shipgen-qa":
        result = run_shipgen_profile_qa(
            args.output_dir,
            ship_classes=args.classes,
            seed=args.seed,
            beam_px=args.beam_px,
            length_px=args.length_px,
            bg_color=args.bg_color,
            trim_mode=_normalize_variant_choice(args.trim_mode),
            visible_side=_normalize_variant_choice(args.visible_side),
        )
        print(f"manifest: {result.manifest_path}")
        print(f"summary: {result.summary_path}")
        if result.offenders:
            for offender in result.offenders:
                print(
                    f"outline offender: {offender.ship_class} left={offender.left_edge_delta:.2f} right={offender.right_edge_delta:.2f}",
                    file=sys.stderr,
                )
            raise SystemExit(1)
        print(f"shipgen QA passed for {len(result.records)} ships")
        return
    if args.command == "shadow-preview":
        outputs = render_shadow_previews(args.output_dir)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    if args.command == "wake-preview":
        outputs = render_wake_previews(args.output_dir)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return


if __name__ == "__main__":
    main()