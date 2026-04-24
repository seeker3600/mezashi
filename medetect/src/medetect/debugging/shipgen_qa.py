"""Ten-ship beam-profile QA for shipgen outputs."""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from medetect.debugging.pixel_profile import (
    extract_line_profile,
    save_profile_visualization,
    write_profile_table,
)
from medetect.debugging.ship_profile import (
    composite_rgba_on_background,
    summarize_profile_values,
    summarize_rendered_ship_profile,
)
from medetect.datagen.render import rasterize_ship_svg
from medetect.shipgen.gen import generate_ship_svg

DEFAULT_SHIPGEN_QA_CLASSES = (
    "amphib_assault",
    "barge",
    "barge_deck",
    "carrier",
    "corvette",
    "destroyer",
    "destroyer_stealth",
    "fishing_longliner",
    "fishing_purse_seiner",
    "tug_harbor",
)


def _extract_trim_metadata(svg_text: str) -> tuple[str, str]:
    root = ET.fromstring(svg_text)
    trim_mode = root.attrib.get("data-trim-mode", "none")
    visible_side = root.attrib.get("data-visible-side", "none")
    return trim_mode, visible_side


@dataclass(frozen=True)
class ShipgenQaRecord:
    ship_class: str
    trim_mode: str
    visible_side: str
    image_path: str
    profile_tsv_path: str
    profile_png_path: str
    x0_px: int
    y0_px: int
    x1_px: int
    y1_px: int
    left_edge_delta: float
    right_edge_delta: float
    has_dark_outline: bool
    has_bright_outline: bool


@dataclass(frozen=True)
class ShipgenQaResult:
    output_dir: str
    manifest_path: str
    summary_path: str
    records: tuple[ShipgenQaRecord, ...]

    @property
    def offenders(self) -> tuple[ShipgenQaRecord, ...]:
        return tuple(record for record in self.records if record.has_dark_outline or record.has_bright_outline)


def run_shipgen_profile_qa(
    output_dir: Path,
    *,
    ship_classes: Sequence[str] = DEFAULT_SHIPGEN_QA_CLASSES,
    seed: int = 42,
    beam_px: int = 128,
    length_px: int = 640,
    bg_color: tuple[int, int, int] = (40, 60, 90),
    hull_noise: float = 0.005,
    deck_scatter_density: float = 3.0,
    trim_mode: str | None = "none",
    visible_side: str | None = "none",
) -> ShipgenQaResult:
    """Render the standard 10-ship QA set and write profile artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    profile_dir = output_dir / "profiles"
    image_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    records: list[ShipgenQaRecord] = []
    manifest_lines = [
        "class\ttrim_mode\tvisible_side\timage\tprofile_tsv\tprofile_png\tx0\ty0\tx1\ty1\tleft_delta\tright_delta\tdark_outline\tbright_outline"
    ]

    for ship_class in ship_classes:
        svg = generate_ship_svg(
            ship_class,
            rng=random.Random(seed),
            hull_noise=hull_noise,
            deck_scatter_density=deck_scatter_density,
            trim_mode=trim_mode,
            visible_side=visible_side,
        )
        resolved_trim_mode, resolved_visible_side = _extract_trim_metadata(svg)
        rgba = rasterize_ship_svg(svg, beam_px, length_px)
        rgb = composite_rgba_on_background(rgba, bg_color=bg_color)
        image_path = image_dir / f"{ship_class}.png"
        Image.fromarray(rgb).save(image_path)

        metrics = summarize_rendered_ship_profile(rgba, bg_color=bg_color)
        positions, values = extract_line_profile(
            rgb,
            metrics.x0_px,
            metrics.y_px,
            metrics.x1_px,
            metrics.y_px,
        )
        outline = summarize_profile_values(values)

        profile_tsv_path = profile_dir / f"{ship_class}_profile.tsv"
        with profile_tsv_path.open("w", encoding="utf-8") as handle:
            write_profile_table(positions, values, handle, print_all=True)

        profile_png_path = profile_dir / f"{ship_class}_profile.png"
        save_profile_visualization(
            rgb,
            positions,
            values,
            (metrics.x0_px, metrics.y_px),
            (metrics.x1_px, metrics.y_px),
            profile_png_path,
        )

        record = ShipgenQaRecord(
            ship_class=ship_class,
            trim_mode=resolved_trim_mode,
            visible_side=resolved_visible_side,
            image_path=image_path.as_posix(),
            profile_tsv_path=profile_tsv_path.as_posix(),
            profile_png_path=profile_png_path.as_posix(),
            x0_px=metrics.x0_px,
            y0_px=metrics.y_px,
            x1_px=metrics.x1_px,
            y1_px=metrics.y_px,
            left_edge_delta=outline.left_edge_delta,
            right_edge_delta=outline.right_edge_delta,
            has_dark_outline=outline.has_dark_outline,
            has_bright_outline=outline.has_bright_outline,
        )
        records.append(record)
        manifest_lines.append(
            "\t".join(
                [
                    record.ship_class,
                    record.trim_mode,
                    record.visible_side,
                    record.image_path,
                    record.profile_tsv_path,
                    record.profile_png_path,
                    str(record.x0_px),
                    str(record.y0_px),
                    str(record.x1_px),
                    str(record.y1_px),
                    f"{record.left_edge_delta:.2f}",
                    f"{record.right_edge_delta:.2f}",
                    str(record.has_dark_outline),
                    str(record.has_bright_outline),
                ]
            )
        )

    manifest_path = output_dir / "manifest.tsv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "records": [asdict(record) for record in records],
                "offenders": [record.ship_class for record in records if record.has_dark_outline or record.has_bright_outline],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ShipgenQaResult(
        output_dir=output_dir.as_posix(),
        manifest_path=manifest_path.as_posix(),
        summary_path=summary_path.as_posix(),
        records=tuple(records),
    )
