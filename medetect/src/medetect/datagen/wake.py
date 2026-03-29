"""Ship wake trail rendering for synthetic satellite imagery.

Each ship is assigned a random motion state (STOPPED / SLOW / MEDIUM / FAST).
Ships receive a wake trail with a probability that depends on their state.
Wakes are rendered as two main elements:

1. **Stern foam** — a short, bright elliptical patch directly behind the stern
   (the brightest part, always lighter than the sea surface).
2. **Trailing line** — a thin centre-line extending far astern, optionally
   accompanied by very faint diverging side-lines.

Three visual patterns are selected per-ship:

- **Foam only** (~25 %): short foam patch, no trailing line (harbour / slow).
- **Foam + trail** (~50 %): foam plus a single thin trailing line.
- **Foam + trail + spread** (~25 %): trail with faint ±3°–10° side-lines.

Wake colour adapts to the local sea surface: foam is always brighter, but the
trailing line may be slightly brighter *or* slightly darker than the
background (50 / 50), matching real satellite imagery.

Coordinate convention
---------------------
``angle_rad`` follows the same convention as
:func:`medetect.datagen.compose.compute_obb_corners`: at ``angle_rad = 0``
the bow points upward (−Y).  The wake exits from the stern in direction
``(−sin θ, +cos θ)`` in image-space (Y-axis pointing down).
"""

from __future__ import annotations

import enum
import math
import random

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter


# ── Motion state & wake pattern ───────────────────────────────────────────


class MotionState(enum.Enum):
    """Qualitative motion state for a single ship."""

    STOPPED = "stopped"
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


class WakePattern(enum.Enum):
    """Visual pattern selected for one ship's wake."""

    FOAM_ONLY = "foam_only"
    FOAM_TRAIL = "foam_trail"
    FOAM_TRAIL_SPREAD = "foam_trail_spread"


_STATE_VALUES: list[MotionState] = list(MotionState)
_PATTERN_VALUES: list[WakePattern] = list(WakePattern)

# Sampling weights for the four states.
_STATE_WEIGHTS: list[float] = [0.20, 0.40, 0.30, 0.10]

# Probability of generating a visible wake for each state.
_WAKE_PROB: dict[MotionState, float] = {
    MotionState.STOPPED: 0.20,
    MotionState.SLOW: 0.50,
    MotionState.MEDIUM: 0.80,
    MotionState.FAST: 0.90,
}

# Pattern selection weights per state [foam_only, foam_trail, spread].
# Global mix ≈ 30 % foam-only, 48 % trail, 22 % spread.
_PATTERN_WEIGHTS: dict[MotionState, list[float]] = {
    MotionState.STOPPED: [0.70, 0.25, 0.05],
    MotionState.SLOW: [0.30, 0.55, 0.15],
    MotionState.MEDIUM: [0.10, 0.55, 0.35],
    MotionState.FAST: [0.05, 0.45, 0.50],
}

# Trailing-line length as a multiple of the ship's pixel length.
_TRAIL_LENGTH_FACTOR: dict[MotionState, tuple[float, float]] = {
    MotionState.STOPPED: (0.5, 1.5),
    MotionState.SLOW: (1.5, 3.0),
    MotionState.MEDIUM: (2.5, 5.0),
    MotionState.FAST: (4.0, 8.0),
}

# Maximum alpha for the foam patch.
_FOAM_ALPHA_MAX: dict[MotionState, float] = {
    MotionState.STOPPED: 0.30,
    MotionState.SLOW: 0.50,
    MotionState.MEDIUM: 0.65,
    MotionState.FAST: 0.75,
}

# Trail alpha expressed as a fraction of the foam alpha.
_TRAIL_ALPHA_RATIO: tuple[float, float] = (0.40, 0.70)


def pick_motion_state(rng: random.Random) -> MotionState:
    """Return a random :class:`MotionState` for one ship."""
    return rng.choices(_STATE_VALUES, weights=_STATE_WEIGHTS, k=1)[0]


def _pick_pattern(state: MotionState, rng: random.Random) -> WakePattern:
    """Select a visual wake pattern based on the ship's motion state."""
    return rng.choices(_PATTERN_VALUES, weights=_PATTERN_WEIGHTS[state], k=1)[0]


# ── Colour helpers ────────────────────────────────────────────────────────


def _sample_water_color(
    background: NDArray[np.uint8],
    x: float,
    y: float,
    radius: int = 12,
) -> NDArray[np.float32]:
    """Return the mean RGB colour in a square window centred on *(x, y)*."""
    h, w = background.shape[:2]
    y0 = max(0, int(y) - radius)
    y1 = min(h, int(y) + radius)
    x0 = max(0, int(x) - radius)
    x1 = min(w, int(x) + radius)
    patch = background[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([50.0, 60.0, 70.0], dtype=np.float32)
    return patch.mean(axis=(0, 1)).astype(np.float32)


def _make_wake_color(
    water_color: NDArray[np.float32],
    rng: random.Random,
) -> NDArray[np.float32]:
    """Derive foam colour: water brightened and slightly desaturated.

    Returns an RGB float32 array.  The result is never pure white—it
    retains a hint of the local water hue.
    """
    brightness_boost = rng.uniform(20.0, 40.0)
    desat = rng.uniform(0.10, 0.25)
    mean_brightness = float(water_color.mean())
    wake = water_color + brightness_boost
    # Partial desaturation: nudge toward the mean
    wake = wake * (1.0 - desat) + mean_brightness * desat
    return np.clip(wake, 0.0, 255.0).astype(np.float32)


def _make_trail_color(
    water_color: NDArray[np.float32],
    rng: random.Random,
    darker: bool = False,
) -> NDArray[np.float32]:
    """Derive trailing-line colour: subtle shift from local water.

    50 % of the time the trail is slightly brighter; the other 50 % it is
    slightly darker—matching real satellite imagery where wake trails are
    not uniformly white.
    """
    if darker:
        delta = rng.uniform(-25.0, -6.0)
    else:
        delta = rng.uniform(6.0, 28.0)
    desat = rng.uniform(0.03, 0.10)
    mean_b = float(water_color.mean())
    trail = water_color + delta
    trail = trail * (1.0 - desat) + (mean_b + delta) * desat
    return np.clip(trail, 0.0, 255.0).astype(np.float32)


# ── Noise ─────────────────────────────────────────────────────────────────


def _build_noise_field(
    shape: tuple[int, int],
    rng: random.Random,
    downsample: int = 8,
) -> NDArray[np.float32]:
    """Return a float32 noise field in ``[0.55, 1.0]``.

    Kept for backward-compatible tests; not used by the current renderer.
    """
    seed = rng.randint(0, 2**31 - 1)
    np_rng = np.random.default_rng(seed)
    H, W = shape
    sh = max(4, H // downsample)
    sw = max(4, W // downsample)
    raw = np_rng.random((sh, sw)).astype(np.float32)
    img = Image.fromarray((raw * 255).astype(np.uint8), mode="L")
    img = img.resize((W, H), Image.BILINEAR)
    noise = np.asarray(img).astype(np.float32) / 255.0
    lo, hi = float(noise.min()), float(noise.max())
    if hi > lo:
        noise = (noise - lo) / (hi - lo)
    return 0.55 + 0.45 * noise


def _build_1d_noise(
    n_steps: int,
    rng: random.Random,
) -> NDArray[np.float32]:
    """Low-frequency 1-D noise in ``[0.15, 1.0]`` for trail irregularity.

    About 10–30 % of the trail will be near-transparent, but never fully
    zero—preventing a dotted-line appearance.
    """
    seed = rng.randint(0, 2**31 - 1)
    np_rng = np.random.default_rng(seed)
    n_ctrl = max(4, n_steps // 10)
    ctrl = np_rng.random(n_ctrl).astype(np.float32)
    x_ctrl = np.linspace(0.0, 1.0, n_ctrl)
    x_full = np.linspace(0.0, 1.0, n_steps)
    noise = np.interp(x_full, x_ctrl, ctrl)
    return (0.15 + 0.85 * noise).astype(np.float32)


# ── Path computation ──────────────────────────────────────────────────────


def _compute_path(
    sx: float,
    sy: float,
    dx: float,
    dy: float,
    perp_dx: float,
    perp_dy: float,
    length: float,
    curvature: float,
    n_steps: int,
) -> list[tuple[float, float]]:
    """Return *n_steps + 1* points along a gently curved wake path.

    *curvature* is the total lateral pixel offset at the far end (t = 1).
    Offset grows as t² for a smooth arc.
    """
    pts: list[tuple[float, float]] = []
    for i in range(n_steps + 1):
        t = i / n_steps
        d = t * length
        lat = curvature * t * t
        pts.append((sx + dx * d + perp_dx * lat, sy + dy * d + perp_dy * lat))
    return pts


# ── Element renderers ─────────────────────────────────────────────────────


def _render_foam(
    background: NDArray[np.uint8],
    water_mask: NDArray[np.bool_],
    foam_cx: float,
    foam_cy: float,
    foam_half_l: float,
    foam_half_w: float,
    angle_rad: float,
    foam_color: NDArray[np.float32],
    foam_alpha_max: float,
    blur_radius: float,
) -> None:
    """Render a short bright foam patch behind the stern (in-place)."""
    H, W = background.shape[:2]
    extent = max(foam_half_l, foam_half_w) * 2.5 + int(blur_radius * 3) + 4
    y0 = max(0, int(foam_cy - extent))
    y1 = min(H, int(foam_cy + extent) + 1)
    x0 = max(0, int(foam_cx - extent))
    x1 = min(W, int(foam_cx + extent) + 1)
    if y1 <= y0 or x1 <= x0:
        return
    if foam_half_l < 0.3 or foam_half_w < 0.3:
        return

    rh, rw = y1 - y0, x1 - x0
    ys, xs = np.mgrid[0:rh, 0:rw].astype(np.float32)
    lx = xs + np.float32(x0 - foam_cx)
    ly = ys + np.float32(y0 - foam_cy)

    # Project onto wake-aligned axes.
    wdx = np.float32(-math.sin(angle_rad))
    wdy = np.float32(math.cos(angle_rad))
    pdx = np.float32(math.cos(angle_rad))
    pdy = np.float32(math.sin(angle_rad))

    along = lx * wdx + ly * wdy
    across = lx * pdx + ly * pdy

    hl = np.float32(foam_half_l)
    hw = np.float32(foam_half_w)
    r2 = (along / hl) ** 2 + (across / hw) ** 2
    foam_alpha = np.exp(np.float32(-2.0) * r2).astype(np.float32)

    # Light blur.
    if blur_radius > 0.2:
        u8 = np.clip(foam_alpha * 255, 0, 255).astype(np.uint8)
        pil = Image.fromarray(u8, "L")
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        foam_alpha = np.asarray(pil).astype(np.float32) / 255.0

    # Water mask + scale.
    foam_alpha *= water_mask[y0:y1, x0:x1].astype(np.float32)
    foam_alpha = np.clip(foam_alpha * foam_alpha_max, 0.0, 1.0)

    bm = foam_alpha > 0.003
    if not bm.any():
        return
    region = background[y0:y1, x0:x1]
    a = foam_alpha[bm, np.newaxis]
    blended = region[bm].astype(np.float32) * (1.0 - a) + foam_color * a
    region[bm] = np.clip(blended, 0, 255).astype(np.uint8)


def _render_trail(
    background: NDArray[np.uint8],
    water_mask: NDArray[np.bool_],
    points: list[tuple[float, float]],
    trail_width: int,
    trail_color: NDArray[np.float32],
    trail_alpha_max: float,
    rng: random.Random,
    blur_radius: float,
) -> None:
    """Render a thin trailing line with distance-decay and noise (in-place)."""
    H, W = background.shape[:2]
    n_seg = len(points) - 1
    if n_seg < 1 or trail_alpha_max < 0.003:
        return

    # Bounding box of path with padding.
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = trail_width + int(blur_radius * 3) + 3
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(W, int(max(xs)) + pad + 1)
    y1 = min(H, int(max(ys)) + pad + 1)
    rw, rh = x1 - x0, y1 - y0
    if rw < 2 or rh < 2:
        return

    noise = _build_1d_noise(n_seg, rng)

    trail_img = Image.new("L", (rw, rh), 0)
    draw = ImageDraw.Draw(trail_img)
    for i in range(n_seg):
        t = i / n_seg
        decay = (1.0 - t) ** 0.7
        fill = max(0, min(255, int(decay * float(noise[i]) * 255)))
        if fill < 1:
            continue
        p0 = (round(points[i][0]) - x0, round(points[i][1]) - y0)
        p1 = (round(points[i + 1][0]) - x0, round(points[i + 1][1]) - y0)
        draw.line([p0, p1], fill=fill, width=trail_width)

    trail_alpha = np.asarray(trail_img).astype(np.float32) / 255.0

    if blur_radius > 0.2:
        u8 = np.clip(trail_alpha * 255, 0, 255).astype(np.uint8)
        pil = Image.fromarray(u8, "L")
        pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        trail_alpha = np.asarray(pil).astype(np.float32) / 255.0

    # Water mask + scale.
    trail_alpha *= water_mask[y0:y1, x0:x1].astype(np.float32)
    trail_alpha *= trail_alpha_max

    bm = trail_alpha > 0.002
    if not bm.any():
        return
    region = background[y0:y1, x0:x1]
    a = trail_alpha[bm, np.newaxis]
    blended = region[bm].astype(np.float32) * (1.0 - a) + trail_color * a
    region[bm] = np.clip(blended, 0, 255).astype(np.uint8)


# ── Main entry ────────────────────────────────────────────────────────────


def render_wake(
    background: NDArray[np.uint8],
    water_mask: NDArray[np.bool_],
    cx: float,
    cy: float,
    beam_px: int,
    length_px: int,
    angle_rad: float,
    state: MotionState,
    rng: random.Random,
    *,
    wake_prob_scale: float = 1.0,
    wake_alpha_scale: float = 1.0,
) -> None:
    """Render a ship wake trail onto *background* in-place.

    The wake is rendered **before** the ship silhouette is composited on top
    so that the hull occludes its own wake (correct z-ordering).

    This function is a no-op when:
    - ``wake_prob_scale <= 0`` or ``wake_alpha_scale <= 0``,
    - the probability roll for *state* fails, or
    - the wake region falls entirely outside the image.

    Parameters
    ----------
    background:
        RGB image array ``(H × W × 3)``, modified in place.
    water_mask:
        Boolean mask ``(H × W)``, ``True`` = water.  Wake pixels over land
        are suppressed.
    cx, cy:
        Ship centre in image pixels.
    beam_px:
        Ship beam (width) in pixels.
    length_px:
        Ship length in pixels.
    angle_rad:
        Ship heading in radians (same convention as
        :func:`medetect.datagen.compose.compute_obb_corners`).
        At ``angle_rad = 0`` the bow faces upward (−Y).
    state:
        Motion state; governs wake probability and dimensions.
    rng:
        Random number generator.
    wake_prob_scale:
        Multiplier applied to each state's built-in wake probability before
        the random roll.

        Built-in probabilities per state:
          - STOPPED  20 %
          - SLOW     50 %
          - MEDIUM   80 %
          - FAST     90 %

        ``0.0`` = never generate wakes (all states disabled).
        ``1.0`` = use built-in table as-is (default).
        ``0.5`` = halve every probability (e.g. FAST → 45 %).
        ``2.0`` = double every probability, capped at 100 % each.

        Values above 1.0 are allowed; each per-state result is clamped to
        ``[0, 1]`` so no probability can exceed 100 %.
    wake_alpha_scale:
        Opacity multiplier applied to the rendered wake pixels.
        Controls **how strongly** a wake appears when it is generated.

        ``0.0`` = fully transparent (same effect as disabling wakes).
        ``1.0`` = default strength (default).
        ``1.5`` = 50 % brighter / more opaque wakes.

        This is independent of *wake_prob_scale*: a ship's wake can be
        likely to occur (high ``wake_prob_scale``) but subtle when it does
        (low ``wake_alpha_scale``), or vice-versa.
    """
    if wake_prob_scale <= 0.0 or wake_alpha_scale <= 0.0:
        return
    effective_prob = min(1.0, _WAKE_PROB[state] * wake_prob_scale)
    if rng.random() > effective_prob:
        return

    H, W = background.shape[:2]

    # ── Pattern selection ─────────────────────────────────────────────
    pattern = _pick_pattern(state, rng)

    # ── Directions ────────────────────────────────────────────────────
    wake_dx = -math.sin(angle_rad)
    wake_dy = math.cos(angle_rad)
    perp_dx = math.cos(angle_rad)
    perp_dy = math.sin(angle_rad)

    # Stern position (half the ship length back from centre).
    stern_frac = rng.uniform(0.45, 0.50)
    sx = cx + wake_dx * stern_frac * length_px
    sy = cy + wake_dy * stern_frac * length_px

    # Local water colour.
    water_color = _sample_water_color(background, sx, sy)

    # ── Foam ──────────────────────────────────────────────────────────
    foam_offset = rng.uniform(0.05, 0.15) * length_px
    foam_cx = sx + wake_dx * foam_offset
    foam_cy = sy + wake_dy * foam_offset
    foam_half_l = rng.uniform(0.15, 0.50) * length_px / 2.0
    foam_half_w = rng.uniform(0.4, 1.2) * beam_px / 2.0
    foam_color = _make_wake_color(water_color, rng)
    foam_alpha = _FOAM_ALPHA_MAX[state] * rng.uniform(0.80, 1.0) * wake_alpha_scale
    foam_blur = max(0.4, beam_px * 0.08)

    # ── Trail (patterns 1 & 2) ────────────────────────────────────────
    has_trail = pattern != WakePattern.FOAM_ONLY
    trail_length = 0.0
    if has_trail:
        trail_start_d = foam_offset + rng.uniform(0.05, 0.25) * length_px
        tsi_x = sx + wake_dx * trail_start_d
        tsi_y = sy + wake_dy * trail_start_d

        lo_f, hi_f = _TRAIL_LENGTH_FACTOR[state]
        trail_length = rng.uniform(lo_f, hi_f) * length_px
        trail_length = min(trail_length, math.hypot(H, W) * 0.70)

        trail_width = max(1, min(4, round(beam_px * 0.20)))
        trail_is_dark = rng.random() < 0.5
        trail_color = _make_trail_color(water_color, rng, darker=trail_is_dark)
        trail_alpha = foam_alpha * rng.uniform(*_TRAIL_ALPHA_RATIO)
        trail_blur = max(0.3, beam_px * 0.05)

        # Optional curvature (50 % straight, 50 % gentle arc).
        if rng.random() < 0.5:
            curvature = rng.choice([-1, 1]) * trail_length * rng.uniform(0.01, 0.05)
        else:
            curvature = 0.0

        n_steps = max(20, int(trail_length / 4))
        main_path = _compute_path(
            tsi_x, tsi_y, wake_dx, wake_dy, perp_dx, perp_dy,
            trail_length, curvature, n_steps,
        )

    # ── Render trail first (lower visual priority) ────────────────────
    if has_trail and trail_length >= 2.0:
        _render_trail(
            background, water_mask, main_path, trail_width,
            trail_color, trail_alpha, rng, trail_blur,
        )

        # Side-lines for pattern 2 (very faint diverging auxiliary lines).
        if pattern == WakePattern.FOAM_TRAIL_SPREAD:
            side_alpha = trail_alpha * rng.uniform(0.20, 0.50)
            angle_r = math.radians(rng.uniform(3.0, 10.0))
            angle_l = math.radians(rng.uniform(3.0, 10.0))
            len_r = trail_length * rng.uniform(0.40, 0.80)
            len_l = trail_length * rng.uniform(0.40, 0.80)

            # Right side-line: rotate wake direction toward +perp.
            cos_r, sin_r = math.cos(angle_r), math.sin(angle_r)
            r_dx = wake_dx * cos_r + perp_dx * sin_r
            r_dy = wake_dy * cos_r + perp_dy * sin_r
            path_r = _compute_path(
                tsi_x, tsi_y, r_dx, r_dy, perp_dx, perp_dy,
                len_r, 0.0, max(10, int(len_r / 4)),
            )
            _render_trail(
                background, water_mask, path_r, 1,
                trail_color, side_alpha, rng, trail_blur,
            )

            # Left side-line: rotate wake direction toward −perp.
            cos_l, sin_l = math.cos(angle_l), math.sin(angle_l)
            l_dx = wake_dx * cos_l - perp_dx * sin_l
            l_dy = wake_dy * cos_l - perp_dy * sin_l
            path_l = _compute_path(
                tsi_x, tsi_y, l_dx, l_dy, perp_dx, perp_dy,
                len_l, 0.0, max(10, int(len_l / 4)),
            )
            _render_trail(
                background, water_mask, path_l, 1,
                trail_color, side_alpha, rng, trail_blur,
            )

    # ── Render foam last (highest visual priority) ────────────────────
    _render_foam(
        background, water_mask, foam_cx, foam_cy,
        foam_half_l, foam_half_w, angle_rad, foam_color,
        foam_alpha, foam_blur,
    )
