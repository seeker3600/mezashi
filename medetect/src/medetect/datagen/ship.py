"""Ship sizing and SVG selection helpers for synthetic datagen."""

from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

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
# Inner-band multiplier: SVGs whose L/B ratio falls within
# [natural / M, natural * M] receive weight 1.0 during selection.
# Raised from 1.6 to 1.9 so that slender vessels (e.g. patrol craft,
# 50 m × 7 m => L/B ≈ 7.1) remain inside the full-weight band.
_MAX_REASONABLE_LB_RATIO_MULTIPLIER = 1.9
# Outer-band hard-reject multiplier.  Ratios outside
# [natural / _LB_OUTER_BAND_MULTIPLIER, natural * _LB_OUTER_BAND_MULTIPLIER]
# are scored 0 and never used.  The ratio _LB_OUTER_BAND_MULTIPLIER /
# _MAX_REASONABLE_LB_RATIO_MULTIPLIER is kept ≈ 1.26 (same as before).
_LB_OUTER_BAND_MULTIPLIER = 2.4
def _normalize_lb_ratio_range(
    lb_ratio_range: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if lb_ratio_range is None:
        return None
    lo, hi = lb_ratio_range
    return (min(lo, hi), max(lo, hi))


def _lb_ratio_in_range(
    lb_ratio: float,
    lb_ratio_range: tuple[float, float] | None,
) -> bool:
    if lb_ratio_range is None:
        return True
    lo, hi = _normalize_lb_ratio_range(lb_ratio_range)
    return lo <= lb_ratio <= hi


def _constrained_lb_ratio(
    lb_ratio: float,
    length_m: float,
    lb_ratio_range: tuple[float, float] | None,
) -> float:
    if lb_ratio_range is not None:
        lo, hi = _normalize_lb_ratio_range(lb_ratio_range)
        return min(max(lb_ratio, lo), hi)
    return _effective_lb_ratio(lb_ratio, length_m)


def compute_ship_pixel_size(
    ship_class: str,
    lb_ratio: float,
    resolution_m: float,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
    lb_ratio_range: tuple[float, float] | None = None,
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
    lb_ratio_range
        Optional global ``(min_lb, max_lb)`` hard constraint for the final
        length-to-beam ratio.
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
    effective_lb_ratio = _constrained_lb_ratio(lb_ratio, length_m, lb_ratio_range)
    beam_m = length_m / effective_lb_ratio

    length_px = max(MIN_SHIP_LENGTH_PX, round(length_m / resolution_m))
    beam_px = max(MIN_SHIP_BEAM_PX, round(beam_m / resolution_m))
    return beam_px, length_px


@lru_cache(maxsize=512)
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


def _min_reasonable_lb_ratio(length_m: float) -> float:
    """Return the widest acceptable L/B ratio for a sampled ship length.

    Symmetric counterpart to *_max_reasonable_lb_ratio*: vessels shorter than
    their length would imply get down-weighted during SVG selection.
    """
    return _natural_lb_ratio(length_m) / _MAX_REASONABLE_LB_RATIO_MULTIPLIER


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

    The acceptable band is
    [natural / _MAX_REASONABLE_LB_RATIO_MULTIPLIER,
     natural * _MAX_REASONABLE_LB_RATIO_MULTIPLIER];
    ratios inside score 1.0.  Outside this band the weight decays
    exponentially.  Hard-reject thresholds are symmetric: weight 0.0 below
    natural / _LB_OUTER_BAND_MULTIPLIER or above
    natural * _LB_OUTER_BAND_MULTIPLIER.
    """
    natural = _natural_lb_ratio(target_length_m)
    if lb_ratio > natural * _LB_OUTER_BAND_MULTIPLIER or lb_ratio < natural / _LB_OUTER_BAND_MULTIPLIER:
        return 0.0
    upper_excess = max(0.0, lb_ratio - natural * _MAX_REASONABLE_LB_RATIO_MULTIPLIER)
    lower_deficit = max(0.0, natural / _MAX_REASONABLE_LB_RATIO_MULTIPLIER - lb_ratio)
    return math.exp(-(upper_excess + lower_deficit))


@lru_cache(maxsize=64)
def _shipgen_class_weights(
    target_m: float | None,
    lb_ratio_range: tuple[float, float] | None = None,
) -> tuple[tuple[str, ...], tuple[float, ...] | None]:
    from medetect.shipgen.gen import get_ship_classes
    from medetect.shipgen.ship_class import SHIP_CLASSES

    classes = tuple(get_ship_classes())
    if target_m is None:
        if lb_ratio_range is None:
            return classes, None
        weights = tuple(
            1.0 if _lb_ratio_in_range(
                (SHIP_CLASSES[ship_class].lb[0] + SHIP_CLASSES[ship_class].lb[1]) / 2.0,
                lb_ratio_range,
            ) else 0.0
            for ship_class in classes
        )
        return classes, weights
    weights = tuple(
        (
            _svg_lb_weight(
                (SHIP_CLASSES[ship_class].lb[0] + SHIP_CLASSES[ship_class].lb[1]) / 2.0,
                target_m,
            )
            if _lb_ratio_in_range(
                (SHIP_CLASSES[ship_class].lb[0] + SHIP_CLASSES[ship_class].lb[1]) / 2.0,
                lb_ratio_range,
            )
            else 0.0
        )
        for ship_class in classes
    )
    return classes, weights


def _pick_svg(
    svg_metas: list[_SvgMeta] | None,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
    lb_ratio_range: tuple[float, float] | None = None,
    offnadir_deg: float = 0.0,
    sensor_az_ship_deg: float = 0.0,
    shipgen_kwargs: dict[str, Any] | None = None,
) -> str:
    """Return SVG text weighted by lb_ratio suitability for *length_range*."""
    from medetect.shipgen.gen import generate_ship_svg

    target_m: float | None = (
        (length_range[0] + length_range[1]) / 2.0 if length_range is not None else None
    )

    if svg_metas:
        candidate_metas = [
            meta for meta in svg_metas
            if _lb_ratio_in_range(meta.lb_ratio, lb_ratio_range)
        ]
        if not candidate_metas:
            msg = "No SVG ship variants satisfy the requested ship_lb_ratio range"
            raise ValueError(msg)
        if target_m is not None:
            weights = [_svg_lb_weight(m.lb_ratio, target_m) for m in candidate_metas]
            if not any(weight > 0.0 for weight in weights):
                msg = "No SVG ship variants satisfy both ship_length and ship_lb_ratio constraints"
                raise ValueError(msg)
            (meta,) = rng.choices(candidate_metas, weights=weights, k=1)
        else:
            meta = rng.choice(candidate_metas)
        return _load_svg(meta.path)

    classes, weights = _shipgen_class_weights(target_m, lb_ratio_range)
    if weights is not None and not any(weight > 0.0 for weight in weights):
        msg = "No ship classes satisfy the requested ship_lb_ratio range"
        raise ValueError(msg)

    for _ in range(64):
        if weights is not None:
            (cls,) = rng.choices(classes, weights=weights, k=1)
        else:
            cls = rng.choice(classes)
        svg_text = generate_ship_svg(
            cls,
            rng=rng,
            offnadir_deg=offnadir_deg,
            sensor_az_ship_deg=sensor_az_ship_deg,
            **(shipgen_kwargs or {}),
        )
        _ship_class, generated_lb_ratio = parse_svg_metadata(svg_text)
        if _lb_ratio_in_range(generated_lb_ratio, lb_ratio_range):
            return svg_text

    msg = "Unable to generate a ship variant satisfying the requested ship_lb_ratio range"
    raise ValueError(msg)


def _resolve_ship_dimensions(
    svg_text: str,
    resolution_m: float,
    rng: random.Random,
    length_range: tuple[float, float] | None = None,
    lb_ratio_range: tuple[float, float] | None = None,
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
        lb_ratio_range,
        length_exponent,
    )
    return ship_class, beam_px, length_px, lb_ratio


def _solo_class_names(thresholds: tuple[float, ...]) -> list[str]:
    """Return YOLO class names for standalone (non-tight-cluster) ships.

    Rules:
    - 0 thresholds → ["ship"]
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


def _size_class_names(thresholds: tuple[float, ...]) -> list[str]:
    """Return full YOLO class name list including tight-cluster variants.

    Solo classes come first, followed by ``<name>_c`` variants for
    raft-tight cluster ships.  The *cluster offset* is always
    ``len(solo_names)``, so cluster class IDs = solo_id + len(solo_names).
    """
    solo = _solo_class_names(thresholds)
    return solo + [f"{n}_c" for n in solo]


def _ship_class_id(
    length_px: int,
    resolution_m: float,
    class_id: int,
    size_thresholds: tuple[float, ...] | None,
    *,
    is_cluster: bool = False,
) -> int:
    """Return the YOLO class ID for a ship based on its physical length.

    When *is_cluster* is True the ID is offset by the number of solo classes
    to select the corresponding ``<name>_c`` cluster variant.
    """
    thresholds = size_thresholds or ()
    n_solo = len(thresholds) + 1  # number of solo size buckets
    if thresholds:
        length_m = length_px * resolution_m
        sorted_t = sorted(thresholds)
        bucket = sum(1 for t in sorted_t if length_m >= t)
    else:
        bucket = 0
    cid = class_id + bucket
    if is_cluster:
        cid += n_solo
    return cid