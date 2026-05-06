"""Ship sizing and SVG selection helpers for synthetic datagen."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import NamedTuple

from medetect.datagen.svg import parse_svg_metadata


SHIP_LENGTHS_M: dict[str, tuple[float, float]] = {
    "patrol": (30.0, 80.0),
    "corvette": (80.0, 110.0),
    "frigate": (110.0, 150.0),
    "destroyer": (150.0, 190.0),
    "destroyer_stealth": (140.0, 180.0),
    "carrier": (260.0, 340.0),
    "amphib_assault": (200.0, 260.0),
    "lst_lpd": (120.0, 200.0),
    "supply": (150.0, 210.0),
    "fishing_squid_jigger": (20.0, 50.0),
    "fishing_trawler": (15.0, 40.0),
    "fishing_purse_seiner": (25.0, 60.0),
    "fishing_longliner": (20.0, 45.0),
}

_DEFAULT_LENGTH_M = (30.0, 100.0)
MIN_SHIP_LENGTH_PX = 3
MIN_SHIP_BEAM_PX = 3
_MAX_REASONABLE_LB_RATIO_MULTIPLIER = 1.6


def compute_ship_pixel_size(
    ship_class: str,
    lb_ratio: float,
    resolution_m: float,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
) -> tuple[int, int]:
    """Compute ship raster size ``(beam_px, length_px)`` for the tile resolution.

    A random length within the real-world range for *ship_class* is chosen,
    then converted to pixels at the given *resolution_m*.

    Parameters
    ----------
    length_range
        Global ``(min_m, max_m)`` clamp applied on top of the per-class range.
        When *None*, only the per-class range is used.
    length_exponent
        Controls the size-frequency distribution.  ``1.0`` = log-uniform
        (default, equal probability per multiplicative factor).  ``> 1.0``
        produces more small ships; ``< 1.0`` (towards 0) gives a more
        uniform distribution.
    """
    lo, hi = SHIP_LENGTHS_M.get(ship_class, _DEFAULT_LENGTH_M)
    if length_range is not None:
        lo = max(lo, length_range[0])
        hi = min(hi, length_range[1])
        if lo > hi:
            lo, hi = length_range[0], length_range[1]
    lo = max(lo, 1.0)
    u = rng.random()
    t = u ** length_exponent
    length_m = lo * (hi / lo) ** t
    beam_m = length_m / _effective_lb_ratio(lb_ratio, length_m)

    length_px = max(MIN_SHIP_LENGTH_PX, round(length_m / resolution_m))
    beam_px = max(MIN_SHIP_BEAM_PX, round(beam_m / resolution_m))
    return beam_px, length_px


def _load_svg(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class _SvgMeta(NamedTuple):
    """Pre-read metadata for one SVG ship file."""

    path: Path
    lb_ratio: float


def _load_svg_metas(svg_files: list[Path]) -> list[_SvgMeta]:
    """Read lb_ratio from every SVG file up-front for weighted selection."""
    metas: list[_SvgMeta] = []
    for path in svg_files:
        _cls, lb = parse_svg_metadata(path.read_text(encoding="utf-8"))
        metas.append(_SvgMeta(path=path, lb_ratio=lb))
    return metas


def _natural_lb_ratio(length_m: float) -> float:
    """Empirical L/B ratio typical for a vessel of the given length.

    Derived from real-world data: smaller vessels are proportionally
    wider (lower L/B) than large ships.
    Linear approximation: lb ≈ 3.0 + 0.03 × length_m, capped at 10.
    """
    return min(10.0, 3.0 + 0.03 * length_m)


def _max_reasonable_lb_ratio(length_m: float) -> float:
    """Return the slenderest acceptable L/B ratio for a sampled ship length."""
    return _natural_lb_ratio(length_m) * _MAX_REASONABLE_LB_RATIO_MULTIPLIER


def _effective_lb_ratio(lb_ratio: float, length_m: float) -> float:
    """Clamp an SVG L/B ratio to the sane upper bound for the sampled length."""
    sane_upper = _max_reasonable_lb_ratio(length_m)
    return min(max(lb_ratio, 1e-6), sane_upper)


def _scale_ship_pixel_size(beam_px: int, length_px: int, scale: float) -> tuple[int, int]:
    """Scale ship raster dimensions while preserving shared module minima."""
    return (
        max(MIN_SHIP_BEAM_PX, round(beam_px * scale)),
        max(MIN_SHIP_LENGTH_PX, round(length_px * scale)),
    )


def _svg_lb_weight(lb_ratio: float, target_length_m: float) -> float:
    """Preference weight for an SVG with *lb_ratio* at *target_length_m*.

    lb_ratios within 1.6× the natural value for that length score 1.0.
    Those between 1.6× and 2.0× are steeply down-weighted.
    Those exceeding 2.0× natural are hard-rejected (weight 0.0).
    """
    natural = _natural_lb_ratio(target_length_m)
    if lb_ratio > natural * 2.0:
        return 0.0
    excess = max(0.0, lb_ratio - natural * 1.6)
    return math.exp(-excess / 1.0)


def _pick_svg(
    svg_metas: list[_SvgMeta] | None,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
) -> str:
    """Return SVG text weighted by lb_ratio suitability for *length_range*."""
    target_m: float | None = (
        (length_range[0] + length_range[1]) / 2.0 if length_range is not None else None
    )

    if svg_metas:
        if target_m is not None:
            weights = [_svg_lb_weight(m.lb_ratio, target_m) for m in svg_metas]
            (meta,) = rng.choices(svg_metas, weights=weights, k=1)
        else:
            meta = rng.choice(svg_metas)
        return _load_svg(meta.path)

    from medetect.shipgen.gen import generate_ship_svg, get_ship_classes
    from medetect.shipgen.ship_class import SHIP_CLASSES

    classes = get_ship_classes()
    if target_m is not None:
        weights = [
            _svg_lb_weight(
                (SHIP_CLASSES[c].lb[0] + SHIP_CLASSES[c].lb[1]) / 2.0,
                target_m,
            )
            for c in classes
        ]
        (cls,) = rng.choices(classes, weights=weights, k=1)
    else:
        cls = rng.choice(classes)
    return generate_ship_svg(cls, rng=rng)


def _resolve_ship_dimensions(
    svg_text: str,
    resolution_m: float,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
    length_exponent: float = 1.0,
) -> tuple[str, int, int, float]:
    """Return ``(class_name, beam_px, length_px, lb_ratio)`` without rasterizing."""
    ship_class, lb_ratio = parse_svg_metadata(svg_text)
    beam_px, length_px = compute_ship_pixel_size(
        ship_class,
        lb_ratio,
        resolution_m,
        rng,
        length_range,
        length_exponent,
    )
    return ship_class, beam_px, length_px, lb_ratio


def _size_class_names(thresholds: tuple[float, ...]) -> list[str]:
    """Return YOLO class name list for a given set of size thresholds.

    Rules:
    - 1 threshold  → ["ship_small", "ship_large"]
    - 2 thresholds → ["ship_small", "ship_medium", "ship_large"]
    - 3+ thresholds → ["ship_small", "ship_{T0}_{T1}", ..., "ship_large"]
    """
    n = len(thresholds)
    if n == 0:
        return ["ship"]
    if n == 1:
        return ["ship_small", "ship_large"]
    if n == 2:
        return ["ship_small", "ship_medium", "ship_large"]
    # 3+ thresholds: name intermediate buckets by their boundary values
    names = ["ship_small"]
    sorted_t = sorted(thresholds)
    for i in range(len(sorted_t) - 1):
        lo = int(sorted_t[i])
        hi = int(sorted_t[i + 1])
        names.append(f"ship_{lo}_{hi}")
    names.append("ship_large")
    return names


def _ship_class_id(
    length_px: int,
    resolution_m: float,
    class_id: int,
    size_thresholds: tuple[float, ...] | None,
) -> int:
    """Return the YOLO class ID for a ship based on its physical length."""
    if size_thresholds is None or len(size_thresholds) == 0:
        return class_id
    length_m = length_px * resolution_m
    sorted_t = sorted(size_thresholds)
    # bucket index = number of thresholds the length exceeds
    bucket = sum(1 for t in sorted_t if length_m >= t)
    return class_id + bucket