"""Ship class definitions, colour palettes, and the class registry.

Colour palettes are based on real-world ship paint schemes visible in
high-resolution aerial / satellite imagery.  Military vessels use variants
of haze gray; civilian vessels span a wide range — white, blue, red, dark,
and earthy tones that are clearly distinguishable at 0.3 m/px resolution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

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
        (204, 206, 208),  # Sun-faded white
        (196, 198, 200),  # Cool light gray
        (188, 190, 194),  # Light neutral gray
        (178, 180, 182),  # Weathered light gray
        (164, 166, 170),  # Harbor gray
        (104, 106, 110),  # Medium gray
        (86, 88, 92),     # Charcoal gray
        (56, 58, 62),     # Dark charcoal
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
        (138, 138, 134),  # Faded gray workboat
        (96, 98, 102),    # Steel gray
        (72, 74, 78),     # Charcoal steel
        (48, 52, 56),     # Dark steel
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


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _mix_rgb(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(
        _clamp(int(round((1.0 - amount) * lc + amount * rc)))
        for lc, rc in zip(left, right, strict=True)
    )


def _lift_rgb(rgb: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(_clamp(channel + amount) for channel in rgb)


def _desaturate_toward_gray(
    rgb: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    gray = int(round(sum(rgb) / 3.0))
    return _mix_rgb(rgb, (gray, gray, gray), amount)


def _jitter_rgb(
    rgb: tuple[int, int, int],
    rng: random.Random,
    amount: int,
) -> tuple[int, int, int]:
    return tuple(_clamp(channel + rng.randint(-amount, amount)) for channel in rgb)


def _cap_luminance(
    rgb: tuple[int, int, int],
    max_luminance: float,
) -> tuple[int, int, int]:
    lum = _luminance(rgb)
    if lum <= max_luminance:
        return rgb
    scale = max_luminance / max(lum, 1.0)
    return tuple(_clamp(int(round(channel * scale))) for channel in rgb)


_STRUCT_NEUTRALS: dict[str, list[tuple[int, int, int]]] = {
    "fishing_mixed": [
        (174, 176, 172),
        (180, 182, 176),
        (170, 176, 184),
        (182, 178, 170),
        (168, 174, 178),
    ],
    "fishing_white": [
        (174, 176, 174),
        (180, 182, 180),
        (186, 188, 184),
        (188, 190, 192),
        (194, 194, 190),
    ],
    "work_mixed": [
        (146, 146, 140),
        (154, 150, 142),
        (162, 156, 146),
        (148, 154, 160),
        (170, 162, 148),
    ],
    "barge_dull": [
        (116, 118, 120),
        (124, 122, 116),
        (132, 128, 120),
        (126, 130, 132),
        (138, 132, 122),
    ],
}


_STRUCT_BASE_LUMINANCE_CAPS: dict[str, float] = {
    "fishing_mixed": 192.0,
    "fishing_white": 198.0,
    "work_mixed": 170.0,
    "barge_dull": 160.0,
}


_STRUCT_MAX_HULL_LUMINANCE_GAPS: dict[str, float] = {
    "fishing_mixed": 52.0,
    "work_mixed": 48.0,
    "barge_dull": 44.0,
}


_SMALL_SHIP_STRUCT_LUMINANCE_CAPS: dict[str, float] = {
    "fishing_white": 184.0,
}


_SMALL_SHIP_STRUCT_HULL_GAPS: dict[str, float] = {
    "fishing_white": 26.0,
}


_STRUCT_FAMILY_ALIASES: dict[str, str] = {
    "blue_variants": "fishing_mixed",
    "red_orange": "work_mixed",
    "brown_tan": "work_mixed",
    "green_variants": "work_mixed",
    "gray_variants": "barge_dull",
}


_TRIM_PRIMARY_WEIGHTS: dict[str, tuple[int, int, int]] = {
    "navy_gray": (82, 8, 10),
    "navy_dark": (86, 6, 8),
    "fishing_mixed": (48, 22, 30),
    "fishing_white": (30, 40, 30),
    "work_mixed": (62, 12, 26),
    "barge_dull": (76, 6, 18),
}


_TRIM_SIDE_WEIGHTS: dict[str, tuple[int, int, int]] = {
    "navy_gray": (48, 26, 26),
    "navy_dark": (52, 24, 24),
    "fishing_mixed": (36, 32, 32),
    "fishing_white": (34, 33, 33),
    "work_mixed": (42, 29, 29),
    "barge_dull": (54, 23, 23),
}


_PERIMETER_TRIM_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "navy_gray": [(214, 216, 220), (222, 222, 216), (205, 210, 216)],
    "navy_dark": [(208, 210, 214), (198, 202, 208), (216, 216, 210)],
    "fishing_mixed": [(228, 228, 222), (220, 222, 224), (214, 216, 212)],
    "fishing_white": [(238, 238, 232), (232, 232, 226), (224, 226, 222)],
    "work_mixed": [(228, 224, 216), (216, 216, 210), (206, 206, 200)],
    "barge_dull": [(220, 214, 204), (208, 204, 196), (196, 198, 194)],
}


_BOW_TRIM_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "navy_gray": [(170, 54, 48), (208, 154, 54), (214, 214, 204)],
    "navy_dark": [(168, 48, 42), (198, 140, 50), (206, 204, 198)],
    "fishing_mixed": [(188, 54, 44), (210, 118, 54), (54, 98, 160), (56, 124, 106)],
    "fishing_white": [(196, 48, 42), (44, 90, 150), (58, 126, 110), (214, 156, 60)],
    "work_mixed": [(202, 88, 42), (182, 58, 38), (214, 164, 60), (214, 214, 204)],
    "barge_dull": [(156, 76, 52), (188, 142, 64), (180, 182, 176)],
}


_SIDE_TRIM_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "navy_gray": [(102, 110, 122), (132, 44, 42), (82, 88, 96)],
    "navy_dark": [(80, 88, 100), (126, 42, 38), (94, 98, 106)],
    "fishing_mixed": [(162, 54, 44), (44, 84, 132), (54, 112, 98), (84, 88, 96)],
    "fishing_white": [(170, 56, 48), (48, 86, 136), (60, 116, 102), (96, 100, 106)],
    "work_mixed": [(182, 70, 42), (206, 118, 60), (70, 76, 86), (120, 116, 108)],
    "barge_dull": [(120, 78, 52), (96, 92, 84), (74, 80, 84), (134, 112, 82)],
}


def _fork_rng(rng: random.Random) -> random.Random:
    forked = random.Random()
    forked.setstate(rng.getstate())
    return forked


def _resolve_trim_family(family: str) -> str:
    return _STRUCT_FAMILY_ALIASES.get(family, family)


def _resolve_primary_trim_mode(
    family_key: str,
    rng: random.Random,
    forced: str | None,
) -> str:
    allowed = {"none", "perimeter", "bow"}
    if forced is not None:
        if forced not in allowed:
            msg = f"Unsupported trim_mode: {forced!r}"
            raise ValueError(msg)
        return forced
    weights = _TRIM_PRIMARY_WEIGHTS.get(family_key, (58, 16, 26))
    return rng.choices(["none", "perimeter", "bow"], weights=weights, k=1)[0]


def _resolve_visible_side(
    family_key: str,
    rng: random.Random,
    forced: str | None,
) -> str:
    allowed = {"none", "port", "starboard"}
    if forced is not None:
        if forced not in allowed:
            msg = f"Unsupported visible_side: {forced!r}"
            raise ValueError(msg)
        return forced
    weights = _TRIM_SIDE_WEIGHTS.get(family_key, (40, 30, 30))
    return rng.choices(["none", "port", "starboard"], weights=weights, k=1)[0]


def _sample_perimeter_trim_color(
    family_key: str,
    hull: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    tone = rng.choice(_PERIMETER_TRIM_PALETTES.get(family_key, _PERIMETER_TRIM_PALETTES["work_mixed"]))
    mixed = _mix_rgb(tone, _desaturate_toward_gray(hull, 0.55), rng.uniform(0.04, 0.14))
    lifted = _lift_rgb(mixed, rng.randint(0, 8))
    return _jitter_rgb(lifted, rng, 4)


def _sample_bow_trim_color(
    family_key: str,
    hull: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    tone = rng.choice(_BOW_TRIM_PALETTES.get(family_key, _BOW_TRIM_PALETTES["work_mixed"]))
    mixed = _mix_rgb(tone, hull, rng.uniform(0.08, 0.22))
    return _jitter_rgb(mixed, rng, 5)


def _sample_side_trim_color(
    family_key: str,
    hull: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    tone = rng.choice(_SIDE_TRIM_PALETTES.get(family_key, _SIDE_TRIM_PALETTES["work_mixed"]))
    mixed = _mix_rgb(tone, hull, rng.uniform(0.10, 0.28))
    capped = _cap_luminance(mixed, _luminance(hull) + 26.0)
    return _jitter_rgb(capped, rng, 5)


@dataclass(frozen=True)
class HullTrimStyle:
    """Appearance rules for hull trim paint and one visible side band."""

    primary_mode: str = "none"
    primary_color: tuple[int, int, int] | None = None
    primary_width: float = 0.0
    bow_extent: float = 0.0
    visible_side: str = "none"
    side_color: tuple[int, int, int] | None = None
    side_width: float = 0.0
    side_start: float = 0.0
    side_end: float = 1.0

    def primary_css(self) -> str | None:
        if self.primary_color is None:
            return None
        r, g, b = self.primary_color
        return f"rgb({r},{g},{b})"

    def side_css(self) -> str | None:
        if self.side_color is None:
            return None
        r, g, b = self.side_color
        return f"rgb({r},{g},{b})"


def _sample_hull_trim_style(
    family: str,
    hull: tuple[int, int, int],
    rng: random.Random,
    *,
    trim_mode: str | None = None,
    visible_side: str | None = None,
) -> HullTrimStyle:
    family_key = _resolve_trim_family(family)
    trim_rng = _fork_rng(rng)

    primary_mode = _resolve_primary_trim_mode(family_key, trim_rng, trim_mode)
    resolved_side = _resolve_visible_side(family_key, trim_rng, visible_side)

    primary_color: tuple[int, int, int] | None = None
    primary_width = 0.0
    bow_extent = 0.0
    if primary_mode == "perimeter":
        primary_color = _sample_perimeter_trim_color(family_key, hull, trim_rng)
        primary_width = trim_rng.uniform(0.018, 0.038)
        bow_extent = trim_rng.uniform(0.04, 0.08)
    elif primary_mode == "bow":
        primary_color = _sample_bow_trim_color(family_key, hull, trim_rng)
        primary_width = trim_rng.uniform(0.022, 0.046)
        bow_extent = trim_rng.uniform(0.10, 0.22)

    side_color: tuple[int, int, int] | None = None
    side_width = 0.0
    side_start = 0.0
    side_end = 1.0
    if resolved_side != "none":
        side_color = _sample_side_trim_color(family_key, hull, trim_rng)
        side_width = trim_rng.uniform(0.028, 0.062)
        side_start = trim_rng.uniform(0.06, 0.18)
        side_end = trim_rng.uniform(0.78, 0.96)

    return HullTrimStyle(
        primary_mode=primary_mode,
        primary_color=primary_color,
        primary_width=primary_width,
        bow_extent=bow_extent,
        visible_side=resolved_side,
        side_color=side_color,
        side_width=side_width,
        side_start=side_start,
        side_end=side_end,
    )


def _sample_civilian_struct_base(
    family: str,
    hull: tuple[int, int, int],
    rng: random.Random,
) -> tuple[int, int, int]:
    family_key = _STRUCT_FAMILY_ALIASES.get(family, family)
    tone = rng.choice(_STRUCT_NEUTRALS.get(family_key, _STRUCT_NEUTRALS["work_mixed"]))
    hull_muted = _desaturate_toward_gray(hull, rng.uniform(0.25, 0.65))
    roll = rng.random()
    struct_cap = _STRUCT_BASE_LUMINANCE_CAPS.get(family_key, 188.0)

    if family_key == "fishing_white":
        if roll < 0.10:
            struct_base = _jitter_rgb(_lift_rgb(tone, rng.randint(18, 26)), rng, 3)
            struct_cap = 214.0
        elif roll < 0.56:
            struct_base = _lift_rgb(_mix_rgb(hull_muted, tone, 0.46), rng.randint(0, 6))
            struct_cap = 198.0
        else:
            struct_base = _lift_rgb(_desaturate_toward_gray(hull, 0.62), rng.randint(-2, 6))
            struct_cap = 196.0
    elif family_key == "fishing_mixed":
        if roll < 0.18:
            struct_base = _jitter_rgb(_mix_rgb(tone, hull_muted, 0.18), rng, 5)
        elif roll < 0.56:
            struct_base = _lift_rgb(_mix_rgb(hull_muted, tone, 0.22), rng.randint(6, 16))
        elif roll < 0.84:
            struct_base = _lift_rgb(_desaturate_toward_gray(hull, 0.35), rng.randint(10, 22))
        else:
            struct_base = _lift_rgb(_mix_rgb(hull, tone, 0.14), rng.randint(4, 12))
    elif family_key == "work_mixed":
        if roll < 0.08:
            struct_base = _jitter_rgb(_mix_rgb(tone, hull_muted, 0.14), rng, 5)
        elif roll < 0.40:
            struct_base = _lift_rgb(_mix_rgb(hull_muted, tone, 0.28), rng.randint(4, 12))
        elif roll < 0.78:
            struct_base = _lift_rgb(_desaturate_toward_gray(hull, 0.28), rng.randint(6, 14))
        else:
            struct_base = _lift_rgb(_mix_rgb(hull, tone, 0.16), rng.randint(2, 10))
    else:
        if roll < 0.05:
            struct_base = _jitter_rgb(_mix_rgb(tone, hull_muted, 0.12), rng, 4)
        elif roll < 0.48:
            struct_base = _lift_rgb(_mix_rgb(hull_muted, tone, 0.24), rng.randint(4, 12))
        elif roll < 0.84:
            struct_base = _lift_rgb(_desaturate_toward_gray(hull, 0.40), rng.randint(4, 14))
        else:
            struct_base = _lift_rgb(_mix_rgb(hull, tone, 0.14), rng.randint(0, 8))

    capped = _cap_luminance(
        struct_base,
        struct_cap,
    )
    max_gap = _STRUCT_MAX_HULL_LUMINANCE_GAPS.get(family_key)
    if max_gap is None:
        return capped
    return _cap_luminance(capped, _luminance(hull) + max_gap)


@dataclass(frozen=True)
class ShipAppearanceVariant:
    """Optional low-probability appearance toggles for one ship instance."""

    small_ship: bool = False
    oversized_struct: bool = False
    bright_white_struct: bool = False


def _sample_bright_white_small_ship_struct(
    rng: random.Random,
) -> tuple[int, int, int]:
    tone = rng.choice(_STRUCT_NEUTRALS["fishing_white"])
    bright = _lift_rgb(_desaturate_toward_gray(tone, 0.70), rng.randint(18, 28))
    return _jitter_rgb(_cap_luminance(bright, 226.0), rng, 2)


def _apply_small_ship_struct_base_variant(
    family: str,
    hull: tuple[int, int, int],
    struct_base: tuple[int, int, int],
    rng: random.Random,
    appearance_variant: ShipAppearanceVariant | None,
) -> tuple[int, int, int]:
    if appearance_variant is None or not appearance_variant.small_ship:
        return struct_base

    if appearance_variant.bright_white_struct:
        return _sample_bright_white_small_ship_struct(_fork_rng(rng))

    family_key = _STRUCT_FAMILY_ALIASES.get(family, family)
    max_luminance = _SMALL_SHIP_STRUCT_LUMINANCE_CAPS.get(family_key)
    if max_luminance is None:
        return struct_base

    toned = _mix_rgb(struct_base, _desaturate_toward_gray(hull, 0.60), 0.32)
    toned = _cap_luminance(toned, max_luminance)
    max_gap = _SMALL_SHIP_STRUCT_HULL_GAPS.get(family_key)
    if max_gap is None:
        return toned
    return _cap_luminance(toned, _luminance(hull) + max_gap)


def sample_ship_appearance_variant(
    ship_class: ShipClass,
    rng: random.Random,
) -> ShipAppearanceVariant:
    """Resolve rare small-ship appearance toggles without advancing the main RNG."""

    has_small_ship_variant = (
        ship_class.rare_oversized_struct_prob > 0.0
        or ship_class.rare_bright_white_struct_prob > 0.0
    )
    if not has_small_ship_variant:
        return ShipAppearanceVariant()

    forked = _fork_rng(rng)
    return ShipAppearanceVariant(
        small_ship=True,
        oversized_struct=forked.random() < ship_class.rare_oversized_struct_prob,
        bright_white_struct=forked.random() < ship_class.rare_bright_white_struct_prob,
    )


@dataclass(frozen=True)
class ShipColors:
    """Resolved colour set for one ship instance."""

    hull: tuple[int, int, int]
    struct_base: tuple[int, int, int]
    trim: HullTrimStyle = field(default_factory=HullTrimStyle)

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


def sample_colors(
    family: str,
    rng: random.Random,
    *,
    trim_mode: str | None = None,
    visible_side: str | None = None,
    appearance_variant: ShipAppearanceVariant | None = None,
) -> ShipColors:
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

    struct_base: tuple[int, int, int]
    if family == "navy_gray":
        boost = rng.randint(18, 34)
        struct_base = (
            _clamp(hull[0] + boost),
            _clamp(hull[1] + boost),
            _clamp(hull[2] + boost),
        )
    elif family == "navy_dark":
        boost = rng.randint(14, 26)
        struct_base = (
            _clamp(hull[0] + boost),
            _clamp(hull[1] + boost),
            _clamp(hull[2] + boost),
        )
    else:
        # Civilian ships keep family-specific wheelhouse palettes instead of
        # defaulting most dark hulls to the same bright white block.
        struct_base = _sample_civilian_struct_base(family, hull, rng)
    struct_base = _apply_small_ship_struct_base_variant(
        family,
        hull,
        struct_base,
        rng,
        appearance_variant,
    )
    trim = _sample_hull_trim_style(
        family,
        hull,
        rng,
        trim_mode=trim_mode,
        visible_side=visible_side,
    )
    return ShipColors(hull=hull, struct_base=struct_base, trim=trim)


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
    debug_only: bool = False
    rare_oversized_struct_prob: float = 0.0
    rare_bright_white_struct_prob: float = 0.0


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
        rare_oversized_struct_prob=0.02,
        rare_bright_white_struct_prob=0.02,
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
        rare_oversized_struct_prob=0.02,
        rare_bright_white_struct_prob=0.02,
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
        rare_oversized_struct_prob=0.02,
        rare_bright_white_struct_prob=0.02,
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
        rare_oversized_struct_prob=0.02,
        rare_bright_white_struct_prob=0.02,
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
        rare_oversized_struct_prob=0.02,
        rare_bright_white_struct_prob=0.02,
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
        rare_oversized_struct_prob=0.02,
        rare_bright_white_struct_prob=0.02,
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
        rare_oversized_struct_prob=0.02,
        rare_bright_white_struct_prob=0.02,
    ),
    "debug_rect": ShipClass(
        hull="box",
        lb=(6.0, 6.0),
        bow=(0.0, 0.0),
        stern_hw=(0.5, 0.5),
        color_family="navy_gray",
        structs=(),
        details=(),
        debug_only=True,
    ),
}
