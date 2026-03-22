"""Ship class definitions, colour palettes, and the class registry.

Colour palettes are based on real-world ship paint schemes observed in
satellite imagery.  Military vessels use variants of haze gray / dark gray;
fishing vessels use a wider range including blue, white, red, and green.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# ── Colour system ────────────────────────────────────────────────────────

_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "navy_gray": [
        (140, 143, 146),  # USN Haze Gray
        (128, 132, 138),  # Medium gray
        (120, 128, 140),  # Blue-gray (European navies)
        (155, 158, 161),  # Light gray
        (110, 115, 120),  # Dark gray
        (135, 140, 148),  # JMSDF blue-gray
        (95, 100, 108),   # Russian Navy dark
    ],
    "navy_dark": [
        (100, 105, 108),  # Non-skid flight deck
        (90, 95, 100),    # Dark deck
        (85, 90, 98),     # Dark blue-gray
        (105, 110, 115),  # Medium-dark
    ],
    "fishing_mixed": [
        (195, 200, 205),  # White
        (70, 110, 160),   # Light blue
        (45, 75, 130),    # Medium blue
        (30, 50, 90),     # Dark blue
        (150, 55, 45),    # Red / rust
        (40, 85, 55),     # Green
        (180, 185, 195),  # Off-white
        (40, 45, 50),     # Black (working vessel)
    ],
    "fishing_white": [
        (195, 200, 205),
        (200, 205, 210),
        (185, 190, 200),
        (190, 195, 198),
    ],
}


def _clamp(v: int) -> int:
    return max(0, min(255, v))


@dataclass(frozen=True)
class ShipColors:
    """Resolved colour set for one ship instance."""

    hull: tuple[int, int, int]
    struct_base: tuple[int, int, int]

    def hull_css(self) -> str:
        r, g, b = self.hull
        return f"rgb({r},{g},{b})"

    def struct_css(
        self,
        brightness_off: int = 0,
        rng: random.Random | None = None,
    ) -> str:
        jitter = rng.randint(-5, 5) if rng else 0
        r = _clamp(self.struct_base[0] + brightness_off + jitter)
        g = _clamp(self.struct_base[1] + brightness_off + jitter)
        b = _clamp(self.struct_base[2] + brightness_off + jitter)
        return f"rgb({r},{g},{b})"

    def detail_css(self, offset: int) -> str:
        r = _clamp(self.hull[0] + offset)
        g = _clamp(self.hull[1] + offset)
        b = _clamp(self.hull[2] + offset)
        return f"rgb({r},{g},{b})"


def sample_colors(family: str, rng: random.Random) -> ShipColors:
    """Sample a colour scheme from the given palette family."""
    base = rng.choice(_PALETTES[family])
    hull = (
        _clamp(base[0] + rng.randint(-8, 8)),
        _clamp(base[1] + rng.randint(-8, 8)),
        _clamp(base[2] + rng.randint(-8, 8)),
    )
    # Fishing vessels frequently have white superstructure on coloured hull
    if family.startswith("fishing") and rng.random() < 0.5:
        sb = _clamp(190 + rng.randint(-10, 15))
        struct_base = (sb, sb + rng.randint(-3, 3), sb + rng.randint(-3, 5))
    else:
        struct_base = hull
    return ShipColors(hull=hull, struct_base=struct_base)


# ── Structural element data classes ──────────────────────────────────────


@dataclass(frozen=True)
class Struct:
    """Superstructure block placement rule.

    Positions are normalised: bow = 0, stern = 1 along ship length;
    across beam: centre = 0.5, port = 0, starboard = 1.
    """

    x0: tuple[float, float]  # start position range along length
    x1: tuple[float, float]  # end position range along length
    w: tuple[float, float]   # width as fraction of beam
    y_off: float = 0.0       # lateral offset from centre (-0.5..0.5)
    brightness_off: int = 30  # colour brightness offset from struct_base
    prob: float = 1.0        # probability of placement


@dataclass(frozen=True)
class Detail:
    """Small detail element placement rule.

    *kind* selects the drawing routine; *x* is along ship length,
    *y* is across beam (0.5 = centre).
    """

    kind: str  # mast|gun|helipad|circle_spot|vls|crane|lamp|line|door|elevator|funnel|radar_dome|ciws|winch|bollard
    x: tuple[float, float]  # position range along length
    y: float = 0.5          # across beam (0 = port, 1 = starboard)
    size: float = 0.03      # relative to ship length
    prob: float = 1.0


@dataclass(frozen=True)
class ShipClass:
    """Complete ship class template."""

    hull: str                       # key into hull.PROFILES
    lb: tuple[float, float]         # length / beam ratio range
    bow: tuple[float, float]        # bow sharpness (0 = blunt, 1 = sharp)
    stern_hw: tuple[float, float]   # stern half-width range
    color_family: str               # key into _PALETTES
    structs: tuple[Struct, ...]
    details: tuple[Detail, ...]


# ── Ship class registry ──────────────────────────────────────────────────

SHIP_CLASSES: dict[str, ShipClass] = {
    # ─── Military: normal warships ───
    "patrol": ShipClass(
        hull="warship",
        lb=(5.5, 10.0),
        bow=(0.55, 0.95),
        stern_hw=(0.03, 0.18),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.22, 0.30), x1=(0.40, 0.50), w=(0.30, 0.55)),
            Struct(
                x0=(0.52, 0.60), x1=(0.64, 0.72), w=(0.25, 0.45),
                brightness_off=20, prob=0.4,
            ),
        ),
        details=(
            Detail("mast", x=(0.38, 0.48), size=0.04),
            Detail("gun", x=(0.06, 0.14), size=0.03),
            Detail("gun", x=(0.70, 0.80), size=0.02, prob=0.3),
            Detail("radar_dome", x=(0.40, 0.48), size=0.025, prob=0.5),
            Detail("ciws", x=(0.55, 0.62), y=0.7, size=0.02, prob=0.25),
            Detail("funnel", x=(0.50, 0.58), size=0.025, prob=0.6),
            Detail("bollard", x=(0.90, 0.95), y=0.3, size=0.01, prob=0.4),
            Detail("bollard", x=(0.90, 0.95), y=0.7, size=0.01, prob=0.4),
        ),
    ),
    "corvette": ShipClass(
        hull="warship",
        lb=(6.5, 9.5),
        bow=(0.55, 0.90),
        stern_hw=(0.08, 0.22),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.22, 0.28), x1=(0.36, 0.44), w=(0.35, 0.60)),
            Struct(
                x0=(0.48, 0.55), x1=(0.62, 0.70), w=(0.30, 0.55),
                brightness_off=20,
            ),
            Struct(
                x0=(0.72, 0.78), x1=(0.80, 0.86), w=(0.25, 0.45),
                brightness_off=15, prob=0.4,
            ),
        ),
        details=(
            Detail("mast", x=(0.38, 0.48), size=0.04),
            Detail("mast", x=(0.62, 0.70), size=0.03, prob=0.6),
            Detail("gun", x=(0.06, 0.14), size=0.03),
            Detail("vls", x=(0.15, 0.22), size=0.04, prob=0.45),
            Detail("funnel", x=(0.56, 0.64), size=0.03, prob=0.7),
            Detail("ciws", x=(0.70, 0.76), y=0.7, size=0.02, prob=0.35),
            Detail("radar_dome", x=(0.40, 0.46), size=0.025, prob=0.5),
            Detail("helipad", x=(0.82, 0.92), size=0.08, prob=0.5),
            Detail("bollard", x=(0.92, 0.96), y=0.3, size=0.01, prob=0.3),
            Detail("bollard", x=(0.92, 0.96), y=0.7, size=0.01, prob=0.3),
        ),
    ),
    "frigate": ShipClass(
        hull="warship",
        lb=(7.0, 11.0),
        bow=(0.65, 0.95),
        stern_hw=(0.10, 0.22),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.20, 0.28), x1=(0.36, 0.44), w=(0.40, 0.65)),
            Struct(
                x0=(0.46, 0.52), x1=(0.58, 0.64), w=(0.35, 0.58),
                brightness_off=25,
            ),
            Struct(
                x0=(0.64, 0.70), x1=(0.72, 0.78), w=(0.30, 0.52),
                brightness_off=20, prob=0.7,
            ),
            Struct(
                x0=(0.78, 0.82), x1=(0.84, 0.88), w=(0.30, 0.50),
                brightness_off=15, prob=0.35,
            ),
        ),
        details=(
            Detail("mast", x=(0.38, 0.46), size=0.045),
            Detail("mast", x=(0.60, 0.66), size=0.035, prob=0.8),
            Detail("gun", x=(0.06, 0.14), size=0.035),
            Detail("vls", x=(0.14, 0.20), size=0.04, prob=0.6),
            Detail("funnel", x=(0.55, 0.62), size=0.035, prob=0.8),
            Detail("funnel", x=(0.68, 0.74), size=0.03, prob=0.35),
            Detail("ciws", x=(0.42, 0.48), y=0.72, size=0.02, prob=0.4),
            Detail("ciws", x=(0.74, 0.80), y=0.72, size=0.02, prob=0.3),
            Detail("radar_dome", x=(0.40, 0.46), size=0.025, prob=0.6),
            Detail("helipad", x=(0.84, 0.94), size=0.08),
            Detail("bollard", x=(0.94, 0.97), y=0.3, size=0.01, prob=0.3),
            Detail("bollard", x=(0.94, 0.97), y=0.7, size=0.01, prob=0.3),
        ),
    ),
    "destroyer": ShipClass(
        hull="warship",
        lb=(7.5, 11.0),
        bow=(0.70, 0.98),
        stern_hw=(0.08, 0.22),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.18, 0.26), x1=(0.34, 0.42), w=(0.45, 0.70)),
            Struct(
                x0=(0.44, 0.50), x1=(0.56, 0.62), w=(0.40, 0.62),
                brightness_off=25,
            ),
            Struct(
                x0=(0.62, 0.68), x1=(0.72, 0.78), w=(0.35, 0.58),
                brightness_off=20,
            ),
            Struct(
                x0=(0.78, 0.82), x1=(0.84, 0.90), w=(0.30, 0.50),
                brightness_off=15, prob=0.4,
            ),
        ),
        details=(
            Detail("mast", x=(0.36, 0.44), size=0.05),
            Detail("mast", x=(0.58, 0.66), size=0.04, prob=0.85),
            Detail("gun", x=(0.05, 0.12), size=0.04),
            Detail("gun", x=(0.72, 0.78), size=0.025, prob=0.25),
            Detail("vls", x=(0.13, 0.19), size=0.05),
            Detail("vls", x=(0.42, 0.48), size=0.04, prob=0.5),
            Detail("funnel", x=(0.54, 0.60), size=0.04, prob=0.85),
            Detail("funnel", x=(0.68, 0.74), size=0.03, prob=0.4),
            Detail("ciws", x=(0.30, 0.36), y=0.72, size=0.025, prob=0.45),
            Detail("ciws", x=(0.74, 0.80), y=0.28, size=0.025, prob=0.35),
            Detail("radar_dome", x=(0.38, 0.44), size=0.03, prob=0.6),
            Detail("radar_dome", x=(0.60, 0.66), size=0.025, prob=0.4),
            Detail("helipad", x=(0.84, 0.94), size=0.09),
            Detail("bollard", x=(0.94, 0.97), y=0.3, size=0.01, prob=0.3),
            Detail("bollard", x=(0.94, 0.97), y=0.7, size=0.01, prob=0.3),
        ),
    ),
    "destroyer_stealth": ShipClass(
        hull="warship_lean",
        lb=(8.0, 11.5),
        bow=(0.80, 0.98),
        stern_hw=(0.06, 0.16),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.20, 0.28), x1=(0.38, 0.46), w=(0.42, 0.65)),
            Struct(
                x0=(0.48, 0.54), x1=(0.60, 0.68), w=(0.38, 0.58),
                brightness_off=20,
            ),
        ),
        details=(
            Detail("mast", x=(0.40, 0.48), size=0.04),
            Detail("gun", x=(0.06, 0.14), size=0.035),
            Detail("vls", x=(0.14, 0.20), size=0.05),
            Detail("vls", x=(0.44, 0.50), size=0.04, prob=0.5),
            Detail("funnel", x=(0.56, 0.64), size=0.035, prob=0.7),
            Detail("radar_dome", x=(0.42, 0.48), size=0.03, prob=0.6),
            Detail("ciws", x=(0.66, 0.72), y=0.7, size=0.02, prob=0.4),
            Detail("helipad", x=(0.82, 0.92), size=0.09, prob=0.9),
        ),
    ),
    # ─── Military: deck-dominated ───
    "carrier": ShipClass(
        hull="carrier",
        lb=(6.0, 9.5),
        bow=(0.20, 0.55),
        stern_hw=(0.22, 0.40),
        color_family="navy_dark",
        structs=(
            Struct(
                x0=(0.35, 0.48), x1=(0.55, 0.68), w=(0.10, 0.20),
                y_off=0.30, brightness_off=35,
            ),
            Struct(
                x0=(0.22, 0.28), x1=(0.30, 0.36), w=(0.06, 0.12),
                y_off=0.32, brightness_off=30, prob=0.3,
            ),
        ),
        details=(
            Detail("line", x=(0.28, 0.42), y=0.3, size=0.50, prob=0.7),
            Detail("line", x=(0.15, 0.22), y=0.35, size=0.30, prob=0.4),
            Detail("elevator", x=(0.24, 0.30), y=0.85, size=0.06, prob=0.8),
            Detail("elevator", x=(0.58, 0.65), y=0.85, size=0.06, prob=0.8),
            Detail("elevator", x=(0.40, 0.48), y=0.12, size=0.06, prob=0.4),
            Detail("circle_spot", x=(0.12, 0.18), size=0.05, prob=0.5),
            Detail("circle_spot", x=(0.70, 0.78), size=0.05, prob=0.5),
            Detail("mast", x=(0.45, 0.55), y=0.85, size=0.03, prob=0.6),
            Detail("radar_dome", x=(0.50, 0.58), y=0.85, size=0.025, prob=0.5),
        ),
    ),
    "amphib_assault": ShipClass(
        hull="carrier",
        lb=(5.5, 8.5),
        bow=(0.18, 0.50),
        stern_hw=(0.24, 0.42),
        color_family="navy_dark",
        structs=(
            Struct(
                x0=(0.28, 0.40), x1=(0.48, 0.58), w=(0.12, 0.24),
                y_off=0.28, brightness_off=35,
            ),
            Struct(
                x0=(0.60, 0.66), x1=(0.68, 0.74), w=(0.08, 0.16),
                y_off=0.30, brightness_off=25, prob=0.3,
            ),
        ),
        details=(
            Detail("circle_spot", x=(0.15, 0.22), size=0.06, prob=0.9),
            Detail("circle_spot", x=(0.30, 0.38), size=0.06, prob=0.9),
            Detail("circle_spot", x=(0.48, 0.55), size=0.06, prob=0.85),
            Detail("circle_spot", x=(0.62, 0.68), size=0.06, prob=0.8),
            Detail("circle_spot", x=(0.74, 0.82), size=0.06, prob=0.7),
            Detail("circle_spot", x=(0.86, 0.92), size=0.06, prob=0.5),
            Detail("door", x=(0.95, 0.98), size=0.04, prob=0.6),
            Detail("elevator", x=(0.35, 0.42), y=0.88, size=0.06, prob=0.6),
            Detail("elevator", x=(0.60, 0.68), y=0.88, size=0.06, prob=0.5),
            Detail("crane", x=(0.70, 0.76), y=0.80, size=0.03, prob=0.3),
            Detail("ciws", x=(0.10, 0.16), y=0.75, size=0.02, prob=0.4),
            Detail("ciws", x=(0.88, 0.94), y=0.75, size=0.02, prob=0.3),
            Detail("radar_dome", x=(0.38, 0.46), y=0.82, size=0.02, prob=0.5),
        ),
    ),
    # ─── Military: landing / transport ───
    "lst_lpd": ShipClass(
        hull="box",
        lb=(4.5, 7.5),
        bow=(0.10, 0.38),
        stern_hw=(0.22, 0.42),
        color_family="navy_gray",
        structs=(
            Struct(
                x0=(0.55, 0.65), x1=(0.72, 0.82), w=(0.35, 0.58),
                brightness_off=30,
            ),
            Struct(
                x0=(0.40, 0.48), x1=(0.52, 0.58), w=(0.25, 0.42),
                brightness_off=20, prob=0.35,
            ),
        ),
        details=(
            Detail("door", x=(0.01, 0.05), size=0.03),
            Detail("door", x=(0.95, 0.98), size=0.04, prob=0.7),
            Detail("helipad", x=(0.82, 0.94), size=0.08, prob=0.8),
            Detail("crane", x=(0.42, 0.50), y=0.75, size=0.03, prob=0.5),
            Detail("crane", x=(0.42, 0.50), y=0.25, size=0.03, prob=0.3),
            Detail("mast", x=(0.68, 0.76), size=0.035, prob=0.7),
            Detail("ciws", x=(0.55, 0.62), y=0.72, size=0.02, prob=0.35),
            Detail("radar_dome", x=(0.70, 0.76), size=0.02, prob=0.4),
            Detail("funnel", x=(0.72, 0.78), size=0.03, prob=0.6),
            Detail("bollard", x=(0.04, 0.08), y=0.25, size=0.01, prob=0.3),
            Detail("bollard", x=(0.04, 0.08), y=0.75, size=0.01, prob=0.3),
        ),
    ),
    "supply": ShipClass(
        hull="tanker",
        lb=(5.5, 9.0),
        bow=(0.15, 0.45),
        stern_hw=(0.12, 0.30),
        color_family="navy_gray",
        structs=(
            Struct(
                x0=(0.50, 0.60), x1=(0.68, 0.78), w=(0.30, 0.52),
                brightness_off=25,
            ),
            Struct(
                x0=(0.20, 0.28), x1=(0.32, 0.38), w=(0.22, 0.40),
                brightness_off=15, prob=0.3,
            ),
        ),
        details=(
            Detail("crane", x=(0.22, 0.30), y=0.65, size=0.04, prob=0.8),
            Detail("crane", x=(0.32, 0.40), y=0.35, size=0.04, prob=0.7),
            Detail("crane", x=(0.42, 0.50), y=0.65, size=0.04, prob=0.5),
            Detail("crane", x=(0.78, 0.85), y=0.35, size=0.04, prob=0.4),
            Detail("funnel", x=(0.62, 0.70), size=0.04, prob=0.8),
            Detail("mast", x=(0.56, 0.64), size=0.04, prob=0.7),
            Detail("helipad", x=(0.85, 0.94), size=0.07, prob=0.6),
            Detail("radar_dome", x=(0.58, 0.64), size=0.02, prob=0.4),
            Detail("bollard", x=(0.15, 0.20), y=0.2, size=0.012, prob=0.4),
            Detail("bollard", x=(0.15, 0.20), y=0.8, size=0.012, prob=0.4),
            Detail("bollard", x=(0.40, 0.45), y=0.2, size=0.012, prob=0.4),
            Detail("bollard", x=(0.40, 0.45), y=0.8, size=0.012, prob=0.4),
        ),
    ),
    # ─── Fishing vessels ───
    "fishing_squid_jigger": ShipClass(
        hull="fishing",
        lb=(4.5, 7.5),
        bow=(0.45, 0.85),
        stern_hw=(0.03, 0.18),
        color_family="fishing_mixed",
        structs=(
            Struct(
                x0=(0.22, 0.32), x1=(0.40, 0.52), w=(0.28, 0.52),
                brightness_off=25,
            ),
            Struct(
                x0=(0.55, 0.62), x1=(0.64, 0.72), w=(0.20, 0.35),
                brightness_off=15, prob=0.3,
            ),
        ),
        details=(
            Detail("mast", x=(0.42, 0.52), size=0.035, prob=0.7),
            Detail("winch", x=(0.72, 0.80), size=0.02, prob=0.5),
            # Lamp rows — port side
            Detail("lamp", x=(0.12, 0.18), y=0.15, size=0.015, prob=0.9),
            Detail("lamp", x=(0.22, 0.28), y=0.15, size=0.015, prob=0.9),
            Detail("lamp", x=(0.32, 0.38), y=0.15, size=0.015, prob=0.9),
            Detail("lamp", x=(0.42, 0.48), y=0.15, size=0.015, prob=0.9),
            Detail("lamp", x=(0.52, 0.58), y=0.15, size=0.015, prob=0.85),
            Detail("lamp", x=(0.62, 0.68), y=0.15, size=0.015, prob=0.75),
            Detail("lamp", x=(0.72, 0.78), y=0.15, size=0.015, prob=0.5),
            # Lamp rows — starboard side
            Detail("lamp", x=(0.12, 0.18), y=0.85, size=0.015, prob=0.9),
            Detail("lamp", x=(0.22, 0.28), y=0.85, size=0.015, prob=0.9),
            Detail("lamp", x=(0.32, 0.38), y=0.85, size=0.015, prob=0.9),
            Detail("lamp", x=(0.42, 0.48), y=0.85, size=0.015, prob=0.9),
            Detail("lamp", x=(0.52, 0.58), y=0.85, size=0.015, prob=0.85),
            Detail("lamp", x=(0.62, 0.68), y=0.85, size=0.015, prob=0.75),
            Detail("lamp", x=(0.72, 0.78), y=0.85, size=0.015, prob=0.5),
            # Centre lamp bar (some jiggers)
            Detail("lamp", x=(0.15, 0.20), y=0.50, size=0.012, prob=0.4),
            Detail("lamp", x=(0.30, 0.35), y=0.50, size=0.012, prob=0.4),
            Detail("lamp", x=(0.50, 0.55), y=0.50, size=0.012, prob=0.35),
        ),
    ),
    "fishing_trawler": ShipClass(
        hull="fishing_wide",
        lb=(3.8, 6.5),
        bow=(0.35, 0.75),
        stern_hw=(0.08, 0.25),
        color_family="fishing_mixed",
        structs=(
            Struct(
                x0=(0.16, 0.25), x1=(0.35, 0.48), w=(0.32, 0.58),
                brightness_off=25,
            ),
            Struct(
                x0=(0.50, 0.58), x1=(0.60, 0.68), w=(0.22, 0.40),
                brightness_off=15, prob=0.35,
            ),
        ),
        details=(
            Detail("crane", x=(0.65, 0.78), y=0.35, size=0.05, prob=0.8),
            Detail("crane", x=(0.65, 0.78), y=0.65, size=0.05, prob=0.6),
            Detail("crane", x=(0.55, 0.62), y=0.35, size=0.04, prob=0.3),
            Detail("mast", x=(0.38, 0.48), size=0.035, prob=0.7),
            Detail("winch", x=(0.78, 0.88), size=0.025, prob=0.6),
            Detail("winch", x=(0.55, 0.62), size=0.02, prob=0.3),
            Detail("funnel", x=(0.40, 0.48), size=0.025, prob=0.55),
            Detail("bollard", x=(0.88, 0.94), y=0.3, size=0.01, prob=0.4),
            Detail("bollard", x=(0.88, 0.94), y=0.7, size=0.01, prob=0.4),
        ),
    ),
    "fishing_purse_seiner": ShipClass(
        hull="fishing_wide",
        lb=(4.5, 7.5),
        bow=(0.30, 0.70),
        stern_hw=(0.08, 0.24),
        color_family="fishing_mixed",
        structs=(
            Struct(
                x0=(0.18, 0.28), x1=(0.36, 0.48), w=(0.28, 0.52),
                brightness_off=25,
            ),
            Struct(
                x0=(0.50, 0.58), x1=(0.62, 0.70), w=(0.18, 0.38),
                y_off=0.15, brightness_off=15, prob=0.55,
            ),
            Struct(
                x0=(0.72, 0.78), x1=(0.80, 0.86), w=(0.15, 0.30),
                brightness_off=10, prob=0.25,
            ),
        ),
        details=(
            Detail("crane", x=(0.58, 0.70), y=0.70, size=0.04, prob=0.7),
            Detail("crane", x=(0.75, 0.85), y=0.65, size=0.035, prob=0.4),
            Detail("mast", x=(0.40, 0.50), size=0.03, prob=0.7),
            Detail("winch", x=(0.82, 0.90), size=0.025, prob=0.6),
            Detail("winch", x=(0.65, 0.72), y=0.35, size=0.02, prob=0.35),
            Detail("funnel", x=(0.42, 0.50), size=0.025, prob=0.5),
            Detail("bollard", x=(0.90, 0.95), y=0.3, size=0.01, prob=0.4),
            Detail("bollard", x=(0.90, 0.95), y=0.7, size=0.01, prob=0.4),
        ),
    ),
    "fishing_longliner": ShipClass(
        hull="fishing",
        lb=(4.5, 8.0),
        bow=(0.45, 0.80),
        stern_hw=(0.03, 0.18),
        color_family="fishing_white",
        structs=(
            Struct(
                x0=(0.18, 0.28), x1=(0.36, 0.48), w=(0.28, 0.52),
                brightness_off=25,
            ),
            Struct(
                x0=(0.52, 0.60), x1=(0.62, 0.70), w=(0.18, 0.35),
                brightness_off=15, prob=0.3,
            ),
        ),
        details=(
            Detail("mast", x=(0.40, 0.50), size=0.03, prob=0.65),
            Detail("crane", x=(0.68, 0.80), y=0.60, size=0.03, prob=0.5),
            Detail("crane", x=(0.68, 0.80), y=0.40, size=0.025, prob=0.25),
            Detail("winch", x=(0.80, 0.88), size=0.02, prob=0.5),
            Detail("funnel", x=(0.38, 0.46), size=0.02, prob=0.45),
            Detail("bollard", x=(0.88, 0.94), y=0.35, size=0.01, prob=0.35),
            Detail("bollard", x=(0.88, 0.94), y=0.65, size=0.01, prob=0.35),
        ),
    ),
}
