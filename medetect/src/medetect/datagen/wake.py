"""Ship wake trail rendering for synthetic satellite imagery.

Each ship is assigned a random motion state (STOPPED / SLOW / MEDIUM / FAST).
Ships receive a wake trail with a probability that depends on their state.
Wakes are rendered as three layers—a central bright stripe, flanking V-diffusion
bands, and a coherent noise field for irregularity—then alpha-composited onto
the background *before* the ship silhouette is blended on top.

Coordinate convention
---------------------
The ``angle_rad`` parameter follows the same convention used in
:func:`medetect.datagen.compose.compute_obb_corners`: at ``angle_rad = 0`` the
ship SVG points upward (−Y, bow at top).  The wake exits from the stern, which
lies in the direction ``(−sin θ, +cos θ)`` in image-space (y-axis pointing
down).

Intentional omissions
---------------------
- **Perlin / Simplex noise**: replaced by bilinear-upscaled random noise
  (no extra dependency; visually sufficient at Sentinel-2 resolution).
- **Sea-roughness detection**: too expensive per-ship; alpha values are kept
  conservative by default instead.
- **Separate harbour logic**: ships are already kept away from land by the
  water-mask erosion in compose.py; STOPPED ships already have a very low
  wake probability (20 %).
- **Per-wake cluster suppression**: clusters are treated as anchored groups,
  so the caller (compose.py) skips wake generation for cluster ships.
"""

from __future__ import annotations

import enum
import math
import random

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter


# ── Motion state ──────────────────────────────────────────────────────────


class MotionState(enum.Enum):
    """Qualitative motion state for a single ship."""

    STOPPED = "stopped"
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


_STATE_VALUES: list[MotionState] = list(MotionState)

# Sampling weights for the four states.
# Expected wake fraction overall:
#   0.20×0.20 + 0.40×0.50 + 0.30×0.80 + 0.10×0.90 ≈ 57 %
# (within the recommended 30–60 % range).
_STATE_WEIGHTS: list[float] = [0.20, 0.40, 0.30, 0.10]

# Probability of generating a visible wake for each state.
_WAKE_PROB: dict[MotionState, float] = {
    MotionState.STOPPED: 0.20,
    MotionState.SLOW: 0.50,
    MotionState.MEDIUM: 0.80,
    MotionState.FAST: 0.90,
}

# Wake length as a multiple of the ship's pixel length.
_LENGTH_FACTOR: dict[MotionState, tuple[float, float]] = {
    MotionState.STOPPED: (0.1, 0.5),
    MotionState.SLOW: (1.0, 2.0),
    MotionState.MEDIUM: (2.0, 4.0),
    MotionState.FAST: (4.0, 6.0),
}

# Maximum composite alpha of the wake at its brightest pixel.
_ALPHA_MAX: dict[MotionState, float] = {
    MotionState.STOPPED: 0.18,
    MotionState.SLOW: 0.38,
    MotionState.MEDIUM: 0.55,
    MotionState.FAST: 0.65,
}


def pick_motion_state(rng: random.Random) -> MotionState:
    """Return a random :class:`MotionState` for one ship."""
    return rng.choices(_STATE_VALUES, weights=_STATE_WEIGHTS, k=1)[0]


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
    """Derive wake colour: water brightened and slightly desaturated.

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


# ── Noise ─────────────────────────────────────────────────────────────────


def _build_noise_field(
    shape: tuple[int, int],
    rng: random.Random,
    downsample: int = 8,
) -> NDArray[np.float32]:
    """Return a float32 noise field in ``[0.55, 1.0]`` for wake irregularity.

    A small random array is generated and bilinearly upscaled to *shape*.
    This efficiently approximates coherent low-frequency noise without
    requiring external noise libraries.
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
    # Map [0, 1] → [0.55, 1.0] so noise never fully extinguishes the wake.
    return 0.55 + 0.45 * noise


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

    # Wake direction: stern side = (−sin θ, +cos θ) in image space.
    wake_dx = -math.sin(angle_rad)
    wake_dy = math.cos(angle_rad)
    # Perpendicular direction (across the beam).
    perp_dx = math.cos(angle_rad)
    perp_dy = math.sin(angle_rad)

    # Wake origin: just behind the stern (0.35–0.45 × full length from centre).
    stern_frac = rng.uniform(0.35, 0.45)
    sx = cx + wake_dx * stern_frac * length_px
    sy = cy + wake_dy * stern_frac * length_px

    # Wake length in pixels.
    lo_f, hi_f = _LENGTH_FACTOR[state]
    wake_length_px = rng.uniform(lo_f, hi_f) * length_px
    # Never extend beyond 70 % of the image diagonal to avoid implausibly long wakes.
    wake_length_px = min(wake_length_px, math.hypot(H, W) * 0.70)
    if wake_length_px < 2.0:
        return

    # Wake half-widths: tapered from stern to far end.
    init_hw = beam_px * rng.uniform(0.15, 0.35)   # half-width at stern
    final_hw = beam_px * rng.uniform(0.8, 2.5)     # half-width at far end
    v_spread = rng.uniform(1.5, 3.0)               # V-band relative to central width

    alpha_max = _ALPHA_MAX[state] * rng.uniform(0.7, 1.0) * wake_alpha_scale

    # Wake colour derived from local water colour.
    water_color = _sample_water_color(background, sx, sy)
    wake_color = _make_wake_color(water_color, rng)  # shape (3,)

    # ── Pixel coordinate grids (full image) ───────────────────────────────
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = xs - np.float32(sx)
    dy = ys - np.float32(sy)

    # Projections onto wake-aligned axes.
    along = dx * np.float32(wake_dx) + dy * np.float32(wake_dy)   # 0 → wake_length
    across = dx * np.float32(perp_dx) + dy * np.float32(perp_dy)  # signed lateral

    in_wake = (along >= 0.0) & (along <= wake_length_px)
    if not in_wake.any():
        return

    # Normalised progress along wake [0, 1].
    t_clip = np.clip(along / wake_length_px, 0.0, 1.0).astype(np.float32)

    # ── Layer 1: Central wake (narrow tapered Gaussian) ───────────────────
    hw_c = np.float32(init_hw) + (np.float32(final_hw) - np.float32(init_hw)) * t_clip
    sigma_c = np.clip(hw_c * 0.45 + 0.5, 0.5, None)
    central = np.exp(-0.5 * (across / sigma_c) ** 2).astype(np.float32)

    # ── Layer 2: V-shape side bands (wider, lower amplitude) ──────────────
    hw_v = hw_c * np.float32(v_spread)
    sigma_v = np.clip(hw_v * 0.55 + 1.0, 1.0, None)
    v_band = (np.exp(-0.5 * (across / sigma_v) ** 2) * 0.35).astype(np.float32)

    # Combine layers.
    wake_alpha = np.clip(central * 0.65 + v_band * 0.35, 0.0, 1.0)

    # ── Along-axis fade envelope ──────────────────────────────────────────
    # Quick fade-in at the stern (first 5 %), constant middle, fade-out in last 30 %.
    fade_in = np.clip(along / (wake_length_px * 0.05 + 0.5), 0.0, 1.0)
    fade_out = np.clip(
        1.0 - (along - wake_length_px * 0.70) / (wake_length_px * 0.30 + 0.5),
        0.0, 1.0,
    )
    wake_alpha = wake_alpha * (fade_in * fade_out).astype(np.float32)
    wake_alpha *= in_wake.astype(np.float32)

    # ── Layer 3: Coherent noise for irregularity ──────────────────────────
    noise = _build_noise_field((H, W), rng)
    wake_alpha *= noise

    # ── Suppress wake pixels over land ───────────────────────────────────
    wake_alpha *= water_mask.astype(np.float32)

    # ── Gaussian blur to soften edges ─────────────────────────────────────
    blur_r = max(0.5, beam_px * 0.15)
    alpha_u8 = np.clip(wake_alpha * 255.0, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha_u8, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=blur_r))
    wake_alpha = np.asarray(alpha_img).astype(np.float32) / 255.0
    # Re-apply water mask: Gaussian blur can bleed alpha into adjacent land pixels.
    wake_alpha *= water_mask.astype(np.float32)

    # Final opacity.
    wake_alpha = np.clip(wake_alpha * alpha_max, 0.0, 1.0)

    # ── Alpha composite onto background ───────────────────────────────────
    alpha3 = wake_alpha[:, :, np.newaxis]
    blended = background.astype(np.float32) * (1.0 - alpha3) + wake_color * alpha3
    background[:] = np.clip(blended, 0, 255).astype(np.uint8)
