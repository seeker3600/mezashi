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

    kind: str  # mast|gun|helipad|circle_spot|vls|crane|lamp|line|door|elevator
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
        lb=(6.0, 9.0),
        bow=(0.6, 0.9),
        stern_hw=(0.05, 0.15),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.22, 0.28), x1=(0.40, 0.48), w=(0.35, 0.55)),
        ),
        details=(
            Detail("mast", x=(0.38, 0.45), size=0.04),
            Detail("gun", x=(0.08, 0.14), size=0.03),
        ),
    ),
    "corvette": ShipClass(
        hull="warship",
        lb=(7.5, 9.0),
        bow=(0.6, 0.85),
        stern_hw=(0.10, 0.20),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.22, 0.28), x1=(0.38, 0.44), w=(0.35, 0.60)),
            Struct(
                x0=(0.48, 0.55), x1=(0.62, 0.68), w=(0.30, 0.50),
                brightness_off=20,
            ),
        ),
        details=(
            Detail("mast", x=(0.40, 0.48), size=0.04),
            Detail("mast", x=(0.62, 0.68), size=0.03, prob=0.6),
            Detail("gun", x=(0.08, 0.14), size=0.03),
            Detail("helipad", x=(0.80, 0.90), size=0.08, prob=0.5),
        ),
    ),
    "frigate": ShipClass(
        hull="warship",
        lb=(8.0, 10.0),
        bow=(0.7, 0.95),
        stern_hw=(0.12, 0.20),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.22, 0.28), x1=(0.38, 0.44), w=(0.40, 0.65)),
            Struct(
                x0=(0.46, 0.52), x1=(0.58, 0.64), w=(0.35, 0.55),
                brightness_off=25,
            ),
            Struct(
                x0=(0.64, 0.70), x1=(0.72, 0.78), w=(0.30, 0.50),
                brightness_off=20, prob=0.7,
            ),
        ),
        details=(
            Detail("mast", x=(0.40, 0.46), size=0.04),
            Detail("mast", x=(0.60, 0.66), size=0.035, prob=0.8),
            Detail("gun", x=(0.08, 0.14), size=0.03),
            Detail("vls", x=(0.15, 0.20), size=0.04, prob=0.6),
            Detail("helipad", x=(0.82, 0.92), size=0.08),
        ),
    ),
    "destroyer": ShipClass(
        hull="warship",
        lb=(8.5, 10.0),
        bow=(0.75, 0.95),
        stern_hw=(0.12, 0.20),
        color_family="navy_gray",
        structs=(
            Struct(x0=(0.20, 0.26), x1=(0.36, 0.42), w=(0.45, 0.70)),
            Struct(
                x0=(0.44, 0.50), x1=(0.56, 0.62), w=(0.40, 0.60),
                brightness_off=25,
            ),
            Struct(
                x0=(0.62, 0.68), x1=(0.72, 0.78), w=(0.35, 0.55),
                brightness_off=20,
            ),
        ),
        details=(
            Detail("mast", x=(0.38, 0.44), size=0.045),
            Detail("mast", x=(0.58, 0.64), size=0.04),
            Detail("gun", x=(0.07, 0.12), size=0.035),
            Detail("vls", x=(0.14, 0.19), size=0.05),
            Detail("vls", x=(0.42, 0.46), size=0.04, prob=0.5),
            Detail("helipad", x=(0.82, 0.92), size=0.09),
        ),
    ),
    # ─── Military: deck-dominated ───
    "carrier": ShipClass(
        hull="carrier",
        lb=(7.0, 9.0),
        bow=(0.3, 0.5),
        stern_hw=(0.25, 0.35),
        color_family="navy_dark",
        structs=(
            Struct(
                x0=(0.35, 0.45), x1=(0.55, 0.65), w=(0.10, 0.18),
                y_off=0.30, brightness_off=35,
            ),
        ),
        details=(
            Detail("line", x=(0.30, 0.40), y=0.3, size=0.50, prob=0.7),
            Detail("elevator", x=(0.25, 0.30), y=0.85, size=0.06, prob=0.8),
            Detail("elevator", x=(0.60, 0.65), y=0.85, size=0.06, prob=0.8),
        ),
    ),
    "amphib_assault": ShipClass(
        hull="carrier",
        lb=(6.0, 8.0),
        bow=(0.25, 0.45),
        stern_hw=(0.28, 0.38),
        color_family="navy_dark",
        structs=(
            Struct(
                x0=(0.30, 0.40), x1=(0.50, 0.58), w=(0.12, 0.22),
                y_off=0.28, brightness_off=35,
            ),
        ),
        details=(
            Detail("circle_spot", x=(0.20, 0.25), size=0.06, prob=0.9),
            Detail("circle_spot", x=(0.40, 0.45), size=0.06, prob=0.9),
            Detail("circle_spot", x=(0.60, 0.65), size=0.06, prob=0.8),
            Detail("circle_spot", x=(0.75, 0.80), size=0.06, prob=0.7),
            Detail("door", x=(0.95, 0.98), size=0.04, prob=0.6),
        ),
    ),
    # ─── Military: landing / transport ───
    "lst_lpd": ShipClass(
        hull="box",
        lb=(5.0, 7.0),
        bow=(0.15, 0.35),
        stern_hw=(0.25, 0.40),
        color_family="navy_gray",
        structs=(
            Struct(
                x0=(0.55, 0.65), x1=(0.72, 0.80), w=(0.35, 0.55),
                brightness_off=30,
            ),
        ),
        details=(
            Detail("door", x=(0.02, 0.05), size=0.03),
            Detail("door", x=(0.95, 0.98), size=0.04, prob=0.7),
            Detail("helipad", x=(0.82, 0.92), size=0.08, prob=0.8),
            Detail("crane", x=(0.45, 0.52), y=0.75, size=0.03, prob=0.5),
        ),
    ),
    "supply": ShipClass(
        hull="box",
        lb=(6.0, 8.0),
        bow=(0.2, 0.4),
        stern_hw=(0.15, 0.25),
        color_family="navy_gray",
        structs=(
            Struct(
                x0=(0.50, 0.58), x1=(0.68, 0.76), w=(0.30, 0.50),
                brightness_off=25,
            ),
        ),
        details=(
            Detail("crane", x=(0.25, 0.32), y=0.65, size=0.04, prob=0.8),
            Detail("crane", x=(0.35, 0.42), y=0.35, size=0.04, prob=0.7),
            Detail("crane", x=(0.78, 0.85), y=0.65, size=0.04, prob=0.5),
            Detail("helipad", x=(0.85, 0.93), size=0.07, prob=0.6),
        ),
    ),
    # ─── Fishing vessels ───
    "fishing_squid_jigger": ShipClass(
        hull="fishing",
        lb=(5.0, 7.0),
        bow=(0.5, 0.8),
        stern_hw=(0.05, 0.15),
        color_family="fishing_mixed",
        structs=(
            Struct(
                x0=(0.25, 0.32), x1=(0.42, 0.50), w=(0.30, 0.50),
                brightness_off=25,
            ),
        ),
        details=(
            # Lamp rows – port side
            Detail("lamp", x=(0.15, 0.20), y=0.15, size=0.015, prob=0.9),
            Detail("lamp", x=(0.25, 0.30), y=0.15, size=0.015, prob=0.9),
            Detail("lamp", x=(0.35, 0.40), y=0.15, size=0.015, prob=0.9),
            Detail("lamp", x=(0.45, 0.50), y=0.15, size=0.015, prob=0.9),
            Detail("lamp", x=(0.55, 0.60), y=0.15, size=0.015, prob=0.8),
            Detail("lamp", x=(0.65, 0.70), y=0.15, size=0.015, prob=0.7),
            # Lamp rows – starboard side
            Detail("lamp", x=(0.15, 0.20), y=0.85, size=0.015, prob=0.9),
            Detail("lamp", x=(0.25, 0.30), y=0.85, size=0.015, prob=0.9),
            Detail("lamp", x=(0.35, 0.40), y=0.85, size=0.015, prob=0.9),
            Detail("lamp", x=(0.45, 0.50), y=0.85, size=0.015, prob=0.9),
            Detail("lamp", x=(0.55, 0.60), y=0.85, size=0.015, prob=0.8),
            Detail("lamp", x=(0.65, 0.70), y=0.85, size=0.015, prob=0.7),
        ),
    ),
    "fishing_trawler": ShipClass(
        hull="fishing_wide",
        lb=(4.5, 6.0),
        bow=(0.4, 0.7),
        stern_hw=(0.10, 0.20),
        color_family="fishing_mixed",
        structs=(
            Struct(
                x0=(0.18, 0.25), x1=(0.38, 0.45), w=(0.35, 0.55),
                brightness_off=25,
            ),
        ),
        details=(
            Detail("crane", x=(0.68, 0.78), y=0.35, size=0.05, prob=0.8),
            Detail("crane", x=(0.68, 0.78), y=0.65, size=0.05, prob=0.6),
            Detail("mast", x=(0.40, 0.48), size=0.035, prob=0.7),
        ),
    ),
    "fishing_purse_seiner": ShipClass(
        hull="fishing_wide",
        lb=(5.0, 7.0),
        bow=(0.4, 0.65),
        stern_hw=(0.10, 0.20),
        color_family="fishing_mixed",
        structs=(
            Struct(
                x0=(0.20, 0.28), x1=(0.38, 0.46), w=(0.30, 0.50),
                brightness_off=25,
            ),
            Struct(
                x0=(0.50, 0.58), x1=(0.62, 0.68), w=(0.20, 0.35),
                y_off=0.15, brightness_off=15, prob=0.6,
            ),
        ),
        details=(
            Detail("crane", x=(0.60, 0.70), y=0.70, size=0.04, prob=0.7),
            Detail("mast", x=(0.42, 0.50), size=0.03, prob=0.7),
        ),
    ),
    "fishing_longliner": ShipClass(
        hull="fishing",
        lb=(5.0, 7.0),
        bow=(0.5, 0.75),
        stern_hw=(0.05, 0.15),
        color_family="fishing_white",
        structs=(
            Struct(
                x0=(0.20, 0.28), x1=(0.38, 0.46), w=(0.30, 0.50),
                brightness_off=25,
            ),
        ),
        details=(
            Detail("mast", x=(0.40, 0.48), size=0.03, prob=0.6),
            Detail("crane", x=(0.70, 0.80), y=0.60, size=0.03, prob=0.5),
        ),
    ),
}
