"""Water mask extraction from Sentinel-2 SCL band.

The Scene Classification Layer (SCL) from Sentinel-2 L2A products
classifies each pixel into land-cover categories.  Value 6 = water.
When SCL data is unavailable, a simple RGB brightness fallback is used.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

# Sentinel-2 L2A SCL class value for water
SCL_WATER = 6


def make_water_mask_from_scl(scl: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Create binary water mask from the SCL band.

    Parameters
    ----------
    scl
        Scene Classification Layer array ``(H, W)`` with ``uint8`` values.
    """
    return scl == SCL_WATER


def make_water_mask_from_rgb(
    rgb: NDArray[np.uint8],
    *,
    brightness_threshold: int = 60,
) -> NDArray[np.bool_]:
    """Heuristic water mask from RGB channels.

    Combines two complementary detectors:

    1. **Dark water** — mean brightness below *brightness_threshold* and
       blue not far below the dominant channel (the original heuristic).
    2. **Bright water** — brightness above the dark threshold but still
       exhibits water-like spectral characteristics:
       blue ≥ red (rules out bare soil / urban),
       not strongly vegetation-green (G-R < 50 and G-B < 40),
       and overall brightness not too high (< 180, excludes clouds /
       bright sand).

    The union of both detectors is returned so that coastal turquoise,
    sediment-laden, and mauve offshore water are captured in addition to
    the classic dark open ocean.

    Parameters
    ----------
    rgb
        RGB array ``(H, W, 3)`` with ``uint8`` values.
    brightness_threshold
        Dark-water brightness ceiling (default 60).
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    brightness: NDArray = rgb.mean(axis=2)

    # --- detector 1: dark water (original) ---
    rg_max = np.maximum(r, g)
    dark_water = (brightness < brightness_threshold) & (b >= rg_max - 10)

    # --- detector 2: bright water ---
    chroma = np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))
    bright_water = (
        (brightness >= brightness_threshold)
        & (brightness < 180)      # exclude clouds / bright sand
        & (b >= r)                # water has blue ≥ red
        & (g - b < 45)           # not strongly green vegetation (G >> B)
        & (r < 160)              # not bright soil/urban
        & (chroma > 15)          # not achromatic grey (urban/concrete)
    )

    return dark_water | bright_water


def erode_mask(
    mask: NDArray[np.bool_],
    pixels: int,
) -> NDArray[np.bool_]:
    """Erode a binary mask to shrink boundaries.

    Uses a square kernel via PIL ``MinFilter`` to keep ships away from
    coastlines and image edges.

    Parameters
    ----------
    mask
        Binary mask ``(H, W)``.
    pixels
        Half-size of the erosion kernel.  0 = no erosion.
    """
    if pixels <= 0:
        return mask
    kernel = 2 * pixels + 1
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.filter(ImageFilter.MinFilter(kernel))
    return np.array(img) > 0
