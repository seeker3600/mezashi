"""Ship class definitions, colour palettes, and the class registry.

Colour palettes are based on real-world ship paint schemes visible in
high-resolution aerial / satellite imagery.  Military vessels use variants
of haze gray; civilian vessels span a wide range — white, blue, red, dark,
and earthy tones that are clearly distinguishable at 0.3 m/px resolution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# ── Colour system ────────────────────────────────────────────────────────

_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    # Military: all variants stay in the blueish-gray band.
    "navy_gray": [
        (140, 143, 146),  # USN Haze Gray
        (128, 132, 138),  # Medium gray
        (124, 130, 138),  # Blue-gray (European navies)
        (150, 152, 156),  # Light gray (coast guard variant)
        (110, 115, 120),  # Dark gray
        (132, 136, 142),  # JMSDF blue-gray
        (98, 102, 108),   # Russian Navy dark
        (118, 120, 128),  # Slate gray
        (145, 147, 153),  # Light slate
        (108, 112, 118),  # Medium-dark gray
    ],
    "navy_dark": [
        (100, 105, 108),  # Non-skid flight deck
        (90, 95, 100),    # Dark deck
        (85, 90, 98),     # Dark blue-gray
        (105, 110, 115),  # Medium-dark
        (75, 80, 90),     # Very dark gray-blue
        (95, 100, 108),   # Medium gray-blue
    ],
    # Fishing: broad real-world palette — white, blue, red, dark, and
    # everything in between.  At 0.3 m/px these colours are clearly visible.
    "fishing_mixed": [
        # White / off-white (most common globally)
        (208, 208, 204),  # Weathered white
        (218, 217, 212),  # Clean white
        (195, 198, 202),  # Bluish off-white
        (225, 225, 220),  # Bright white
        (200, 200, 195),  # Ivory white
        # Blue hulls (common in Asia, Mediterranean, N. Europe)
        (52, 82, 128),    # Medium blue
        (36, 58, 102),    # Deep blue
        (62, 98, 148),    # Cornflower blue
        (28, 46, 86),     # Dark navy blue
        (45, 75, 125),    # Royal blue
        (55, 95, 155),    # Bright blue
        # Red / rust hulls (anti-fouling, traditional)
        (150, 52, 42),    # Hull red
        (132, 72, 50),    # Rust red
        (165, 80, 44),    # Orange-red
        (175, 95, 60),    # Lighter rust
        # Dark hulls
        (42, 44, 50),     # Near-black
        (46, 70, 60),     # Dark green-black
        # Cyan / teal (less common but present)
        (55, 135, 145),   # Teal
        (70, 130, 150),   # Slate blue
    ],
    # Longliners / small white-painted fishing vessels.
    "fishing_white": [
        (218, 218, 214),  # Bright white
        (208, 210, 210),  # Cool white
        (213, 212, 207),  # Warm white
        (200, 203, 202),  # Weathered white
        (225, 222, 218),  # Very bright white
        (210, 215, 210),  # Greenish white
        (220, 215, 220),  # Pinkish white
    ],
    # Work vessels (tugs, pilot boats, workboats, pushers):
    # safety/visibility colours dominate — high-chroma red/orange as well
    # as near-black are both authentic.
    "work_mixed": [
        (178, 60, 36),    # International orange-red (tug classic)
        (192, 86, 30),    # Safety orange
        (158, 46, 40),    # Fire-engine red
        (30, 34, 42),     # Near-black (classic harbour tug)
        (36, 50, 72),     # Dark navy
        (42, 58, 50),     # Dark green (port authority)
        (82, 72, 58),     # Weathered brown-gray
        (112, 108, 104),  # Mid-gray utility vessel
        (148, 144, 138),  # Light gray work vessel
        (200, 110, 60),   # Burnt orange
        (165, 75, 50),    # Darker orange-red
        (55, 65, 85),     # Dark blue-gray
    ],
    # Barges: primarily dark steel, rust, and earthy tones.
    "barge_dull": [
        (68, 64, 58),     # Dark gray-brown
        (55, 52, 50),     # Dark steel
        (92, 76, 56),     # Rust brown
        (108, 78, 50),    # Oxidised orange-brown
        (46, 60, 54),     # Dark green-gray (river barge)
        (86, 84, 80),     # Medium steel gray
        (44, 44, 46),     # Near-black steel
        (78, 62, 50),     # Weathered brown
        (95, 72, 48),     # Tan-rust
        (65, 52, 40),     # Dark chocolate brown
        (72, 68, 62),     # Medium steel
    ],
    # Blue variants — common in Asian fishing fleets
    "blue_variants": [
        (25, 55, 95),     # Deep navy
        (40, 75, 125),    # Royal blue
        (55, 100, 155),   # Bright blue
        (70, 120, 160),   # Light blue
        (35, 70, 115),    # Medium navy
        (45, 85, 140),    # Azure
        (20, 50, 90),     # Very dark blue
        (60, 110, 150),   # Cerulean
    ],
    # Red / orange variants — anti-fouling, safety marking
    "red_orange": [
        (185, 60, 35),    # Bright red
        (165, 85, 45),    # Rust orange
        (200, 70, 40),    # Orange-red
        (145, 55, 35),    # Dark red
        (175, 95, 55),    # Burnt sienna
        (210, 100, 50),   # Safety orange
        (155, 75, 50),    # Reddish-brown
        (190, 85, 60),    # Coral-red
    ],
    # Brown / tan variants — earthy, industrial, aged
    "brown_tan": [
        (115, 95, 75),    # Light brown
        (95, 75, 60),     # Medium brown
        (75, 60, 50),     # Dark brown
        (130, 110, 90),   # Tan
        (145, 120, 95),   # Light tan
        (100, 85, 70),    # Earthy brown
        (85, 68, 55),     # Weathered brown
        (110, 88, 70),    # Russet
    ],
    # Green variants — rare but present (environmental vessels, research)
    "green_variants": [
        (40, 75, 60),     # Dark forest green
        (55, 100, 80),    # Medium green
        (65, 120, 95),    # Sage green
        (45, 85, 70),     # Evergreen
        (50, 95, 75),     # Muted green
        (35, 65, 55),     # Dark teal-green
    ],
    # Gray variants — common utility colours
    "gray_variants": [
        (95, 95, 95),     # Medium gray
        (75, 75, 75),     # Dark gray
        (120, 120, 120),  # Light gray
        (110, 110, 110),  # Medium-light gray
        (85, 85, 85),     # Medium-dark gray
        (100, 100, 100),  # Neutral gray
        (65, 65, 65),     # Very dark gray
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

    def shadow_css(self, rng: random.Random | None = None) -> str:
        """Opaque cast-shadow colour — hull tone darkened with sky-ambient lift.

        Real shadows under diffuse sky illumination are not black: they receive
        scattered blue sky light and retain a faint version of the surface
        colour.  Returns a solid ``rgb(r,g,b)`` string so the ship body stays
        fully opaque; overall ship transparency is set externally by the caller.
        """
        dark = rng.uniform(0.58, 0.72) if rng else 0.65
        jitter = rng.randint(-6, 6) if rng else 0
        r, g, b = self.hull
        sr = _clamp(int(r * dark) + jitter)
        sg = _clamp(int(g * dark) + jitter)
        sb = _clamp(int(b * (dark + 0.08)) + 8 + jitter)  # sky ambient blue lift
        return f"rgb({sr},{sg},{sb})"

    def struct_shadow_css(
        self,
        brightness_off: int = 30,
        rng: random.Random | None = None,
    ) -> str:
        """Opaque self-shadow colour for the shaded face of a superstructure.

        The face turned away from the sun is a darker, slightly cooler version
        of the lit struct face colour.  Returns a solid ``rgb(r,g,b)`` string.
        """
        dark = rng.uniform(0.84, 0.94) if rng else 0.89
        jitter = rng.randint(-4, 4) if rng else 0
        r = _clamp(self.struct_base[0] + brightness_off + jitter)
        g = _clamp(self.struct_base[1] + brightness_off + jitter)
        b = _clamp(self.struct_base[2] + brightness_off + jitter)
        sr = _clamp(int(r * dark))
        sg = _clamp(int(g * dark))
        sb = _clamp(int(b * dark + 4))  # slight blue for sky ambient in shadow
        return f"rgb({sr},{sg},{sb})"


def sample_colors(family: str, rng: random.Random) -> ShipColors:
    """Sample a colour scheme from the given palette family.
    
    Parameters
    ----------
    family
        Palette family name (e.g. 'fishing_mixed', 'navy_gray', 'blue_variants').
    rng
        Random number generator.
    
    Returns
    -------
    ShipColors
        Hull colour and superstructure base colour for the ship instance.
    """
    base = rng.choice(_PALETTES[family])
    hull = (
        _clamp(base[0] + rng.randint(-12, 12)),
        _clamp(base[1] + rng.randint(-12, 12)),
        _clamp(base[2] + rng.randint(-12, 12)),
    )

    # Superstructure / wheelhouse colour strategy:
    #
    # Military:   same gray family, just brighter (uniform hull/struct colouring).
    # Light hull: slight brightness boost — e.g. white hull → pale white struct.
    # Dark / saturated hull: superstructure is almost always white or off-white.
    #   Wheelhouses and accommodation blocks are painted white on virtually all
    #   commercial and fishing vessels regardless of hull colour, both for
    #   reflectivity and as a visibility convention.
    hull_lum = 0.299 * hull[0] + 0.587 * hull[1] + 0.114 * hull[2]

    struct_base: tuple[int, int, int]
    if family in ("navy_gray", "navy_dark"):
        # All-gray military finish.
        boost = rng.randint(18, 38)
        struct_base = (
            _clamp(hull[0] + boost),
            _clamp(hull[1] + boost),
            _clamp(hull[2] + boost),
        )
    elif hull_lum < 130:
        # Dark or strongly saturated hull — white / off-white struct (80 %).
        if rng.random() < 0.80:
            sb = _clamp(195 + rng.randint(-12, 22))
            struct_base = (
                sb,
                sb + rng.randint(-4, 4),
                sb + rng.randint(-4, 6),
            )
        else:
            # Minority case: same hue, strongly boosted.
            boost = rng.randint(30, 58)
            struct_base = (
                _clamp(hull[0] + boost),
                _clamp(hull[1] + boost),
                _clamp(hull[2] + boost),
            )
    else:
        # Light hull — subtle brightness variation.
        if rng.random() < 0.50:
            sb = _clamp(185 + rng.randint(-10, 18))
            struct_base = (
                sb,
                sb + rng.randint(-3, 3),
                sb + rng.randint(-3, 5),
            )
        else:
            boost = rng.randint(10, 28)
            struct_base = (
                _clamp(hull[0] + boost),
                _clamp(hull[1] + boost),
                _clamp(hull[2] + boost),
            )
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

    kind: str  # mast|gun|helipad|circle_spot|vls|crane|lamp|line|door|elevator|funnel|radar_dome|ciws|winch|bollard|shadow|vent|antenna|davit|pipe|liferaft|tire_fender|deck_line|hatch
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
            Detail("shadow", x=(0.38, 0.50), y=0.35, size=0.03, prob=0.5),
            Detail("antenna", x=(0.42, 0.48), size=0.025, prob=0.5),
            Detail("vent", x=(0.52, 0.58), y=0.35, size=0.012, prob=0.35),
            Detail("liferaft", x=(0.64, 0.72), y=0.25, size=0.015, prob=0.35),
            Detail("liferaft", x=(0.64, 0.72), y=0.75, size=0.015, prob=0.35),
            Detail("davit", x=(0.72, 0.78), y=0.25, size=0.02, prob=0.3),
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
            Detail("shadow", x=(0.36, 0.44), y=0.32, size=0.03, prob=0.5),
            Detail("shadow", x=(0.56, 0.64), y=0.32, size=0.025, prob=0.4),
            Detail("antenna", x=(0.40, 0.46), size=0.025, prob=0.5),
            Detail("vent", x=(0.48, 0.54), y=0.35, size=0.012, prob=0.4),
            Detail("liferaft", x=(0.74, 0.80), y=0.25, size=0.016, prob=0.4),
            Detail("liferaft", x=(0.74, 0.80), y=0.75, size=0.016, prob=0.35),
            Detail("davit", x=(0.78, 0.84), y=0.22, size=0.02, prob=0.3),
            Detail("pipe", x=(0.46, 0.56), y=0.22, size=0.01, prob=0.25),
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
            Detail("shadow", x=(0.36, 0.44), y=0.30, size=0.035, prob=0.5),
            Detail("shadow", x=(0.58, 0.64), y=0.30, size=0.03, prob=0.45),
            Detail("antenna", x=(0.40, 0.46), size=0.028, prob=0.55),
            Detail("antenna", x=(0.62, 0.66), size=0.022, prob=0.35),
            Detail("vent", x=(0.46, 0.52), y=0.30, size=0.013, prob=0.4),
            Detail("vent", x=(0.64, 0.70), y=0.70, size=0.013, prob=0.3),
            Detail("liferaft", x=(0.78, 0.84), y=0.24, size=0.016, prob=0.4),
            Detail("liferaft", x=(0.78, 0.84), y=0.76, size=0.016, prob=0.35),
            Detail("davit", x=(0.80, 0.84), y=0.22, size=0.02, prob=0.35),
            Detail("davit", x=(0.80, 0.84), y=0.78, size=0.02, prob=0.3),
            Detail("pipe", x=(0.44, 0.56), y=0.20, size=0.01, prob=0.25),
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
            Detail("shadow", x=(0.34, 0.42), y=0.30, size=0.04, prob=0.55),
            Detail("shadow", x=(0.56, 0.62), y=0.30, size=0.035, prob=0.45),
            Detail("shadow", x=(0.72, 0.78), y=0.30, size=0.03, prob=0.35),
            Detail("antenna", x=(0.38, 0.44), size=0.03, prob=0.55),
            Detail("antenna", x=(0.60, 0.66), size=0.025, prob=0.4),
            Detail("vent", x=(0.44, 0.50), y=0.28, size=0.014, prob=0.4),
            Detail("vent", x=(0.62, 0.68), y=0.72, size=0.014, prob=0.35),
            Detail("liferaft", x=(0.78, 0.84), y=0.22, size=0.016, prob=0.4),
            Detail("liferaft", x=(0.78, 0.84), y=0.78, size=0.016, prob=0.35),
            Detail("davit", x=(0.82, 0.86), y=0.22, size=0.02, prob=0.35),
            Detail("davit", x=(0.82, 0.86), y=0.78, size=0.02, prob=0.3),
            Detail("pipe", x=(0.44, 0.56), y=0.20, size=0.01, prob=0.25),
            Detail("deck_line", x=(0.14, 0.84), y=0.18, size=0.004, prob=0.3),
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
            Detail("shadow", x=(0.38, 0.50), y=0.35, size=0.025, prob=0.5),
            Detail("antenna", x=(0.44, 0.50), size=0.025, prob=0.5),
            Detail("vent", x=(0.55, 0.62), y=0.38, size=0.01, prob=0.35),
            Detail("bollard", x=(0.82, 0.88), y=0.30, size=0.01, prob=0.4),
            Detail("bollard", x=(0.82, 0.88), y=0.70, size=0.01, prob=0.4),
            Detail("pipe", x=(0.60, 0.72), y=0.25, size=0.008, prob=0.3),
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
            Detail("shadow", x=(0.32, 0.46), y=0.33, size=0.03, prob=0.5),
            Detail("antenna", x=(0.40, 0.46), size=0.025, prob=0.45),
            Detail("vent", x=(0.48, 0.54), y=0.36, size=0.012, prob=0.35),
            Detail("liferaft", x=(0.52, 0.60), y=0.24, size=0.015, prob=0.4),
            Detail("davit", x=(0.55, 0.62), y=0.22, size=0.02, prob=0.3),
            Detail("tire_fender", x=(0.85, 0.92), y=0.10, size=0.012, prob=0.35),
            Detail("tire_fender", x=(0.85, 0.92), y=0.90, size=0.012, prob=0.35),
            Detail("hatch", x=(0.15, 0.25), size=0.04, prob=0.3),
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
            Detail("shadow", x=(0.34, 0.46), y=0.34, size=0.028, prob=0.5),
            Detail("antenna", x=(0.42, 0.48), size=0.022, prob=0.45),
            Detail("vent", x=(0.50, 0.58), y=0.35, size=0.012, prob=0.3),
            Detail("liferaft", x=(0.72, 0.80), y=0.25, size=0.015, prob=0.35),
            Detail("davit", x=(0.55, 0.62), y=0.22, size=0.018, prob=0.3),
            Detail("tire_fender", x=(0.88, 0.94), y=0.10, size=0.012, prob=0.3),
            Detail("tire_fender", x=(0.88, 0.94), y=0.90, size=0.012, prob=0.3),
            Detail("pipe", x=(0.60, 0.72), y=0.22, size=0.008, prob=0.25),
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
            Detail("antenna", x=(0.42, 0.48), size=0.025, prob=0.5),
            Detail("liferaft", x=(0.55, 0.62), y=0.25, size=0.02, prob=0.4),
            Detail("vent", x=(0.35, 0.40), y=0.35, size=0.012, prob=0.35),
        ),
    ),
    # ─── Wide / stubby vessels (5–50 m, low L/B) ───
    "tug_harbor": ShipClass(
        hull="tug",
        lb=(2.5, 3.8),
        bow=(0.15, 0.45),
        stern_hw=(0.18, 0.35),
        color_family="work_mixed",
        structs=(
            Struct(
                x0=(0.30, 0.40), x1=(0.55, 0.68), w=(0.40, 0.65),
                brightness_off=30,
            ),
            Struct(
                x0=(0.70, 0.78), x1=(0.80, 0.88), w=(0.30, 0.50),
                brightness_off=20, prob=0.4,
            ),
        ),
        details=(
            Detail("funnel", x=(0.52, 0.62), size=0.05, prob=0.85),
            Detail("mast", x=(0.38, 0.48), size=0.04, prob=0.6),
            Detail("tire_fender", x=(0.25, 0.35), y=0.08, size=0.02, prob=0.8),
            Detail("tire_fender", x=(0.40, 0.50), y=0.08, size=0.02, prob=0.8),
            Detail("tire_fender", x=(0.55, 0.65), y=0.08, size=0.02, prob=0.7),
            Detail("tire_fender", x=(0.25, 0.35), y=0.92, size=0.02, prob=0.8),
            Detail("tire_fender", x=(0.40, 0.50), y=0.92, size=0.02, prob=0.8),
            Detail("tire_fender", x=(0.55, 0.65), y=0.92, size=0.02, prob=0.7),
            Detail("winch", x=(0.12, 0.22), size=0.03, prob=0.6),
            Detail("winch", x=(0.82, 0.90), size=0.025, prob=0.5),
            Detail("bollard", x=(0.08, 0.14), y=0.3, size=0.015, prob=0.5),
            Detail("bollard", x=(0.08, 0.14), y=0.7, size=0.015, prob=0.5),
            Detail("bollard", x=(0.88, 0.94), y=0.3, size=0.015, prob=0.5),
            Detail("bollard", x=(0.88, 0.94), y=0.7, size=0.015, prob=0.5),
            Detail("shadow", x=(0.55, 0.68), y=0.3, size=0.04, prob=0.6),
            Detail("vent", x=(0.70, 0.76), y=0.4, size=0.015, prob=0.4),
            Detail("antenna", x=(0.40, 0.46), size=0.03, prob=0.45),
            Detail("pipe", x=(0.25, 0.55), y=0.25, size=0.015, prob=0.35),
        ),
    ),
    "tug_ocean": ShipClass(
        hull="tug",
        lb=(3.2, 4.8),
        bow=(0.25, 0.55),
        stern_hw=(0.15, 0.30),
        color_family="work_mixed",
        structs=(
            Struct(
                x0=(0.25, 0.35), x1=(0.48, 0.58), w=(0.38, 0.60),
                brightness_off=28,
            ),
            Struct(
                x0=(0.60, 0.68), x1=(0.72, 0.80), w=(0.28, 0.48),
                brightness_off=18, prob=0.5,
            ),
        ),
        details=(
            Detail("funnel", x=(0.48, 0.56), size=0.045, prob=0.85),
            Detail("mast", x=(0.35, 0.44), size=0.04, prob=0.7),
            Detail("tire_fender", x=(0.20, 0.30), y=0.08, size=0.018, prob=0.7),
            Detail("tire_fender", x=(0.35, 0.45), y=0.08, size=0.018, prob=0.7),
            Detail("tire_fender", x=(0.50, 0.60), y=0.08, size=0.018, prob=0.6),
            Detail("tire_fender", x=(0.20, 0.30), y=0.92, size=0.018, prob=0.7),
            Detail("tire_fender", x=(0.35, 0.45), y=0.92, size=0.018, prob=0.7),
            Detail("tire_fender", x=(0.50, 0.60), y=0.92, size=0.018, prob=0.6),
            Detail("winch", x=(0.10, 0.18), size=0.03, prob=0.7),
            Detail("winch", x=(0.80, 0.88), size=0.025, prob=0.5),
            Detail("crane", x=(0.14, 0.22), y=0.65, size=0.035, prob=0.4),
            Detail("bollard", x=(0.06, 0.12), y=0.3, size=0.012, prob=0.5),
            Detail("bollard", x=(0.06, 0.12), y=0.7, size=0.012, prob=0.5),
            Detail("shadow", x=(0.48, 0.58), y=0.35, size=0.035, prob=0.5),
            Detail("vent", x=(0.62, 0.68), y=0.4, size=0.014, prob=0.45),
            Detail("vent", x=(0.62, 0.68), y=0.6, size=0.014, prob=0.35),
            Detail("antenna", x=(0.36, 0.42), size=0.03, prob=0.5),
            Detail("liferaft", x=(0.72, 0.78), y=0.25, size=0.018, prob=0.4),
            Detail("liferaft", x=(0.72, 0.78), y=0.75, size=0.018, prob=0.4),
            Detail("pipe", x=(0.22, 0.48), y=0.22, size=0.012, prob=0.3),
        ),
    ),
    "barge": ShipClass(
        hull="barge",
        lb=(2.5, 4.5),
        bow=(0.05, 0.20),
        stern_hw=(0.25, 0.42),
        color_family="barge_dull",
        structs=(
            Struct(
                x0=(0.80, 0.88), x1=(0.90, 0.96), w=(0.25, 0.42),
                brightness_off=20, prob=0.5,
            ),
        ),
        details=(
            Detail("hatch", x=(0.10, 0.22), size=0.08, prob=0.7),
            Detail("hatch", x=(0.28, 0.40), size=0.08, prob=0.7),
            Detail("hatch", x=(0.46, 0.58), size=0.08, prob=0.6),
            Detail("hatch", x=(0.64, 0.76), size=0.08, prob=0.5),
            Detail("bollard", x=(0.05, 0.10), y=0.15, size=0.015, prob=0.6),
            Detail("bollard", x=(0.05, 0.10), y=0.85, size=0.015, prob=0.6),
            Detail("bollard", x=(0.90, 0.95), y=0.15, size=0.015, prob=0.5),
            Detail("bollard", x=(0.90, 0.95), y=0.85, size=0.015, prob=0.5),
            Detail("deck_line", x=(0.08, 0.78), y=0.20, size=0.005, prob=0.5),
            Detail("deck_line", x=(0.08, 0.78), y=0.80, size=0.005, prob=0.5),
            Detail("pipe", x=(0.06, 0.75), y=0.12, size=0.01, prob=0.35),
        ),
    ),
    "barge_deck": ShipClass(
        hull="barge",
        lb=(3.0, 5.0),
        bow=(0.08, 0.25),
        stern_hw=(0.22, 0.40),
        color_family="barge_dull",
        structs=(
            Struct(
                x0=(0.72, 0.80), x1=(0.86, 0.94), w=(0.32, 0.55),
                brightness_off=25,
            ),
            Struct(
                x0=(0.58, 0.65), x1=(0.68, 0.74), w=(0.20, 0.38),
                brightness_off=15, prob=0.35,
            ),
        ),
        details=(
            Detail("crane", x=(0.22, 0.35), y=0.60, size=0.06, prob=0.7),
            Detail("crane", x=(0.40, 0.55), y=0.40, size=0.06, prob=0.5),
            Detail("hatch", x=(0.10, 0.22), size=0.07, prob=0.5),
            Detail("hatch", x=(0.30, 0.42), size=0.07, prob=0.5),
            Detail("funnel", x=(0.78, 0.86), size=0.035, prob=0.6),
            Detail("mast", x=(0.74, 0.82), size=0.03, prob=0.5),
            Detail("bollard", x=(0.04, 0.08), y=0.15, size=0.015, prob=0.6),
            Detail("bollard", x=(0.04, 0.08), y=0.85, size=0.015, prob=0.6),
            Detail("bollard", x=(0.92, 0.96), y=0.15, size=0.015, prob=0.5),
            Detail("bollard", x=(0.92, 0.96), y=0.85, size=0.015, prob=0.5),
            Detail("shadow", x=(0.72, 0.86), y=0.35, size=0.04, prob=0.5),
            Detail("pipe", x=(0.10, 0.55), y=0.15, size=0.012, prob=0.4),
            Detail("vent", x=(0.68, 0.74), y=0.35, size=0.015, prob=0.35),
            Detail("antenna", x=(0.76, 0.82), size=0.025, prob=0.4),
        ),
    ),
    "pilot_boat": ShipClass(
        hull="launch",
        lb=(3.0, 4.5),
        bow=(0.50, 0.85),
        stern_hw=(0.15, 0.30),
        color_family="work_mixed",
        structs=(
            Struct(
                x0=(0.22, 0.32), x1=(0.52, 0.62), w=(0.45, 0.70),
                brightness_off=30,
            ),
        ),
        details=(
            Detail("mast", x=(0.35, 0.48), size=0.035, prob=0.6),
            Detail("antenna", x=(0.38, 0.48), size=0.03, prob=0.55),
            Detail("radar_dome", x=(0.40, 0.48), size=0.025, prob=0.5),
            Detail("tire_fender", x=(0.25, 0.35), y=0.08, size=0.018, prob=0.65),
            Detail("tire_fender", x=(0.40, 0.50), y=0.08, size=0.018, prob=0.55),
            Detail("tire_fender", x=(0.25, 0.35), y=0.92, size=0.018, prob=0.65),
            Detail("tire_fender", x=(0.40, 0.50), y=0.92, size=0.018, prob=0.55),
            Detail("bollard", x=(0.10, 0.16), y=0.3, size=0.012, prob=0.45),
            Detail("bollard", x=(0.10, 0.16), y=0.7, size=0.012, prob=0.45),
            Detail("shadow", x=(0.38, 0.55), y=0.35, size=0.03, prob=0.5),
            Detail("vent", x=(0.55, 0.62), y=0.35, size=0.012, prob=0.3),
            Detail("liferaft", x=(0.58, 0.65), y=0.30, size=0.018, prob=0.4),
        ),
    ),
    "workboat": ShipClass(
        hull="workboat",
        lb=(2.8, 4.2),
        bow=(0.20, 0.55),
        stern_hw=(0.10, 0.28),
        color_family="work_mixed",
        structs=(
            Struct(
                x0=(0.25, 0.35), x1=(0.50, 0.60), w=(0.40, 0.65),
                brightness_off=28,
            ),
            Struct(
                x0=(0.62, 0.70), x1=(0.74, 0.82), w=(0.25, 0.45),
                brightness_off=18, prob=0.35,
            ),
        ),
        details=(
            Detail("funnel", x=(0.48, 0.56), size=0.04, prob=0.7),
            Detail("mast", x=(0.36, 0.44), size=0.035, prob=0.6),
            Detail("crane", x=(0.62, 0.72), y=0.60, size=0.04, prob=0.5),
            Detail("winch", x=(0.12, 0.20), size=0.025, prob=0.55),
            Detail("winch", x=(0.78, 0.86), size=0.025, prob=0.45),
            Detail("bollard", x=(0.06, 0.12), y=0.25, size=0.012, prob=0.5),
            Detail("bollard", x=(0.06, 0.12), y=0.75, size=0.012, prob=0.5),
            Detail("bollard", x=(0.88, 0.94), y=0.25, size=0.012, prob=0.5),
            Detail("bollard", x=(0.88, 0.94), y=0.75, size=0.012, prob=0.5),
            Detail("tire_fender", x=(0.22, 0.32), y=0.08, size=0.015, prob=0.5),
            Detail("tire_fender", x=(0.22, 0.32), y=0.92, size=0.015, prob=0.5),
            Detail("shadow", x=(0.42, 0.56), y=0.32, size=0.035, prob=0.5),
            Detail("vent", x=(0.55, 0.62), y=0.38, size=0.013, prob=0.4),
            Detail("antenna", x=(0.38, 0.44), size=0.028, prob=0.45),
            Detail("hatch", x=(0.65, 0.75), size=0.05, prob=0.35),
            Detail("pipe", x=(0.20, 0.45), y=0.20, size=0.01, prob=0.3),
            Detail("liferaft", x=(0.62, 0.70), y=0.22, size=0.016, prob=0.35),
        ),
    ),
    "landing_craft": ShipClass(
        hull="barge",
        lb=(3.0, 5.0),
        bow=(0.05, 0.15),
        stern_hw=(0.25, 0.40),
        color_family="navy_gray",
        structs=(
            Struct(
                x0=(0.72, 0.80), x1=(0.86, 0.94), w=(0.28, 0.50),
                brightness_off=25,
            ),
        ),
        details=(
            Detail("door", x=(0.01, 0.04), size=0.03),
            Detail("mast", x=(0.78, 0.86), size=0.035, prob=0.6),
            Detail("funnel", x=(0.82, 0.90), size=0.03, prob=0.7),
            Detail("bollard", x=(0.06, 0.10), y=0.15, size=0.015, prob=0.5),
            Detail("bollard", x=(0.06, 0.10), y=0.85, size=0.015, prob=0.5),
            Detail("bollard", x=(0.92, 0.96), y=0.15, size=0.015, prob=0.5),
            Detail("bollard", x=(0.92, 0.96), y=0.85, size=0.015, prob=0.5),
            Detail("deck_line", x=(0.10, 0.68), y=0.22, size=0.006, prob=0.5),
            Detail("deck_line", x=(0.10, 0.68), y=0.78, size=0.006, prob=0.5),
            Detail("shadow", x=(0.72, 0.86), y=0.35, size=0.04, prob=0.55),
            Detail("antenna", x=(0.80, 0.86), size=0.03, prob=0.4),
            Detail("vent", x=(0.74, 0.80), y=0.38, size=0.012, prob=0.35),
            Detail("hatch", x=(0.25, 0.38), size=0.06, prob=0.4),
            Detail("hatch", x=(0.42, 0.55), size=0.06, prob=0.35),
        ),
    ),
    "pusher": ShipClass(
        hull="tug",
        lb=(2.0, 3.2),
        bow=(0.08, 0.30),
        stern_hw=(0.25, 0.42),
        color_family="work_mixed",
        structs=(
            Struct(
                x0=(0.18, 0.28), x1=(0.42, 0.55), w=(0.45, 0.70),
                brightness_off=28,
            ),
        ),
        details=(
            Detail("funnel", x=(0.38, 0.50), size=0.06, prob=0.8),
            Detail("mast", x=(0.25, 0.35), size=0.04, prob=0.55),
            Detail("tire_fender", x=(0.60, 0.72), y=0.08, size=0.022, prob=0.85),
            Detail("tire_fender", x=(0.76, 0.86), y=0.08, size=0.022, prob=0.7),
            Detail("tire_fender", x=(0.60, 0.72), y=0.92, size=0.022, prob=0.85),
            Detail("tire_fender", x=(0.76, 0.86), y=0.92, size=0.022, prob=0.7),
            Detail("winch", x=(0.58, 0.68), size=0.03, prob=0.5),
            Detail("bollard", x=(0.06, 0.12), y=0.3, size=0.015, prob=0.5),
            Detail("bollard", x=(0.06, 0.12), y=0.7, size=0.015, prob=0.5),
            Detail("bollard", x=(0.88, 0.94), y=0.3, size=0.015, prob=0.6),
            Detail("bollard", x=(0.88, 0.94), y=0.7, size=0.015, prob=0.6),
            Detail("shadow", x=(0.28, 0.45), y=0.32, size=0.04, prob=0.5),
            Detail("vent", x=(0.50, 0.56), y=0.38, size=0.015, prob=0.4),
            Detail("antenna", x=(0.28, 0.34), size=0.035, prob=0.4),
        ),
    ),
}
