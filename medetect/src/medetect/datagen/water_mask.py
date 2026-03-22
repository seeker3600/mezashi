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
    """Fallback water mask from RGB mean brightness.

    Water in Sentinel-2 TCI tends to be dark.  This is a rough heuristic;
    SCL-based masking is strongly preferred.

    Parameters
    ----------
    rgb
        RGB array ``(H, W, 3)`` with ``uint8`` values.
    brightness_threshold
        Pixels with mean brightness below this value are classified as water.
    """
    brightness: NDArray = rgb.mean(axis=2)
    return brightness < brightness_threshold


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
