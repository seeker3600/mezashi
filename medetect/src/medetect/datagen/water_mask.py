"""Water mask extraction from Sentinel-2 SCL band and OSM coastlines.

The Scene Classification Layer (SCL) from Sentinel-2 L2A products
classifies each pixel into land-cover categories.  Value 6 = water.
When SCL data is unavailable, a simple RGB brightness fallback is used.

When an OSM coastline shapefile is available, it provides precise
land/water boundaries.  Coastlines are rasterized onto the tile grid
and the RGB heuristic is used to disambiguate which side is water.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

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


# ── Coastline-based water mask ────────────────────────────────────────────


# SHP shape types that carry a Bounding Box field (Polyline, Polygon, *Z, *M).
_SHP_BBOX_TYPES: frozenset[int] = frozenset({3, 5, 13, 15, 23, 25})


def _read_shp_bboxes(
    shp_path: Path,
) -> tuple[NDArray[np.float64], list[int]]:
    """Read per-record bounding boxes from SHP/SHX without loading coordinates.

    Uses the SHX offset table to seek directly to the 32-byte bbox header of
    each record in the SHP file via ``mmap``.  This reads ~44 bytes per record
    regardless of how many coordinate points it contains, making it fast even
    for a 1.2 GB shapefile.

    Returns
    -------
    bbox_arr
        ``(N, 4)`` array of ``[xmin, ymin, xmax, ymax]`` in the shapefile CRS.
    valid_indices
        Corresponding record indices (0-based) used for ``Reader.shape(i)``.
    """
    import mmap
    import struct

    shx_path = shp_path.with_suffix(".shx")

    # SHX: 100-byte file header + 8 bytes per record (offset, length) big-endian int32
    shx_bytes = shx_path.read_bytes()
    n_records = (len(shx_bytes) - 100) // 8
    shx_recs = np.frombuffer(shx_bytes[100:], dtype=">i4").reshape(n_records, 2)
    offsets_words = shx_recs[:, 0]  # byte offset = value × 2

    bboxes: list[tuple[float, float, float, float]] = []
    valid_indices: list[int] = []

    with open(shp_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for i in range(n_records):
                # Each SHP record: 8-byte record header + content.
                # Content layout for polyline/polygon: shape_type(4) + bbox(32) + …
                content = int(offsets_words[i]) * 2 + 8
                shape_type = struct.unpack_from("<i", mm, content)[0]
                if shape_type not in _SHP_BBOX_TYPES:
                    continue
                xmin, ymin, xmax, ymax = struct.unpack_from("<4d", mm, content + 4)
                if xmin == xmax and ymin == ymax:
                    continue  # degenerate / point-like
                bboxes.append((xmin, ymin, xmax, ymax))
                valid_indices.append(i)
        finally:
            mm.close()

    bbox_arr = np.array(bboxes, dtype=np.float64).reshape(-1, 4)
    return bbox_arr, valid_indices


class CoastlineIndex:
    """Spatial index for OSM coastline geometries (EPSG:4326).

    Memory- and time-efficient design:

    * **Index build**: reads only the 32-byte bbox header of each SHP record
      via ``mmap``, skipping all coordinate data.  For a 1.2 GB shapefile
      this reads ~38 MB of relevant bytes regardless of geometry complexity.
    * **STRtree**: populated with Shapely 2.x ``shapely.box()`` vectorised
      call — no Python-level loop over 870 K objects.
    * **Query**: only the handful of candidate records that hit the tile bbox
      are loaded in full from disk via ``Reader.shape(i)`` (O(1) via .shx).
    * **Per-worker footprint**: ~80 MB (bbox array + STRtree) instead of
      ~500 MB for the naive all-geometries approach.
    """

    def __init__(self, shapefile_path: Path | str) -> None:
        import shapefile as shp
        import shapely
        from shapely import STRtree

        self._path = Path(shapefile_path)
        logger.info("Building coastline bbox index: %s", self._path)

        bbox_arr, valid_indices = _read_shp_bboxes(self._path)
        logger.info("Indexed %d coastline bboxes", len(valid_indices))

        if len(valid_indices) > 0:
            # Shapely 2.x vectorised creation — ~10× faster than looping
            boxes = shapely.box(
                bbox_arr[:, 0], bbox_arr[:, 1],
                bbox_arr[:, 2], bbox_arr[:, 3],
            )
        else:
            boxes = np.array([], dtype=object)

        self._tree = STRtree(boxes)
        self._valid_indices = valid_indices
        # Keep reader open — .shape(i) uses .shx offsets for O(1) random
        # access so we can load individual geometries on demand at query time.
        self._reader = shp.Reader(str(self._path))

    def query(
        self, bounds: tuple[float, float, float, float]
    ) -> list:
        """Return coastline geometries intersecting *bounds*.

        Parameters
        ----------
        bounds
            ``(minx, miny, maxx, maxy)`` in the same CRS as the shapefile
            (EPSG:4326).
        """
        from shapely.geometry import box as shapely_box
        from shapely.geometry import shape as shapely_shape

        hits = self._tree.query(shapely_box(*bounds))
        if len(hits) == 0:
            return []

        geoms = []
        for hit_idx in hits:
            shp_idx = self._valid_indices[int(hit_idx)]
            try:
                s = self._reader.shape(shp_idx)
                geom = shapely_shape(s.__geo_interface__)
                if not geom.is_empty:
                    geoms.append(geom)
            except Exception:  # noqa: BLE001
                pass
        return geoms


def make_water_mask_from_coastline(
    coastline_geoms: list,
    tile_rgb: NDArray[np.uint8],
    transform: object,
    width: int,
    height: int,
) -> NDArray[np.bool_]:
    """Create a water mask using coastline geometries and RGB heuristic.

    Coastline lines are rasterized as boundary pixels.  Connected regions
    separated by these boundaries are then classified as water or land
    using a majority vote from :func:`make_water_mask_from_rgb`.

    Parameters
    ----------
    coastline_geoms
        Shapely geometries (LineStrings) of coastline segments
        intersecting the tile extent.
    tile_rgb
        RGB tile ``(H, W, 3)`` for fallback water heuristic.
    transform
        Affine transform mapping pixel coordinates to geographic
        coordinates (as returned by ``rasterio``).
    width, height
        Tile dimensions in pixels.
    """
    if not coastline_geoms:
        # No coastlines in this tile — assume open ocean.  The caller
        # will AND this with the RGB mask for extra safety.
        return np.ones((height, width), dtype=bool)

    from rasterio.features import rasterize
    from scipy.ndimage import label
    from shapely.geometry import mapping

    # Rasterize coastline lines as 1-pixel-wide boundaries.
    geojson_geoms = [(mapping(g), 1) for g in coastline_geoms]
    burned = rasterize(
        geojson_geoms,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    boundary = burned > 0

    # RGB water heuristic for disambiguation.
    rgb_water = make_water_mask_from_rgb(tile_rgb)

    # Label connected regions separated by coastline boundaries.
    labeled, n_labels = label(~boundary)

    # Majority vote per region: classify as water when >50% of pixels
    # in the region are identified as water by the RGB heuristic.
    flat_labels = labeled.ravel()
    flat_water = rgb_water.ravel().astype(np.intp)

    region_sizes = np.bincount(flat_labels, minlength=n_labels + 1)
    water_counts = np.bincount(
        flat_labels, weights=flat_water, minlength=n_labels + 1,
    )

    is_water = np.zeros(n_labels + 1, dtype=bool)
    valid = region_sizes > 0
    is_water[valid] = water_counts[valid] > region_sizes[valid] * 0.5
    # Label 0 is the boundary itself — not water.
    is_water[0] = False

    return is_water[labeled]
