"""Lightweight SVG metadata helpers for datagen."""

from __future__ import annotations

from functools import lru_cache
import xml.etree.ElementTree as ET


@lru_cache(maxsize=4096)
def parse_svg_metadata(svg_text: str) -> tuple[str, float]:
    """Extract ship class and L/B ratio from SVG attributes.

    Returns
    -------
    tuple[str, float]
        ``(ship_class, lb_ratio)``.  Defaults to ``("unknown", vb_h)``
        when attributes are absent.
    """
    root = ET.fromstring(svg_text)
    ship_class = root.get("data-ship-class", "unknown")
    lb_str = root.get("data-lb-ratio")
    if lb_str is not None:
        lb_ratio = float(lb_str)
    else:
        vb = root.get("viewBox", "0 0 1 1").split()
        lb_ratio = float(vb[3])
    return ship_class, lb_ratio