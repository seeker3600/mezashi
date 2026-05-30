from __future__ import annotations

import numpy as np
import pytest

import medetect.datagen.water_mask as water_mask_mod

from medetect.datagen.water_mask import (
    CoastlineIndex,
    erode_mask,
    make_water_mask_from_coastline,
    make_water_mask_from_rgb,
    make_water_mask_from_scl,
)


class TestMakeWaterMaskFromScl:
    def test_scl_6_is_water(self) -> None:
        """SCL値6（水域）がTrueとなる。"""
        scl = np.array([[6, 4], [5, 6]], dtype=np.uint8)
        mask = make_water_mask_from_scl(scl)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_no_data_excluded(self) -> None:
        """SCL値0（NoData）は水域に含まれない。"""
        scl = np.zeros((3, 3), dtype=np.uint8)
        mask = make_water_mask_from_scl(scl)
        assert not mask.any()

    def test_cloud_excluded(self) -> None:
        """雲のSCL値(7,8,9,10)は水域に含まない。"""
        scl = np.array([7, 8, 9, 10], dtype=np.uint8).reshape(2, 2)
        mask = make_water_mask_from_scl(scl)
        assert not mask.any()

    def test_all_water(self) -> None:
        """全域水域のケース。"""
        scl = np.full((5, 5), 6, dtype=np.uint8)
        mask = make_water_mask_from_scl(scl)
        assert mask.all()


class TestErodeMask:
    def test_no_erosion_returns_same(self) -> None:
        """侵食0のとき元のマスクがそのまま返る。"""
        mask = np.ones((5, 5), dtype=bool)
        result = erode_mask(mask, 0)
        np.testing.assert_array_equal(result, mask)

    def test_erosion_shrinks_boundary(self) -> None:
        """侵食が水域マスクの境界を削る。"""
        mask = np.zeros((9, 9), dtype=bool)
        mask[1:8, 1:8] = True  # Water with land border
        result = erode_mask(mask, 1)
        # Outermost water row should be eroded
        assert not result[1, 1]
        # Centre should remain
        assert result[4, 4]

    def test_small_region_erased(self) -> None:
        """小さな水域は侵食で消える。"""
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True  # Single pixel
        result = erode_mask(mask, 1)
        assert not result.any()


class TestMakeWaterMaskFromRgb:
    def test_dark_pixels_classified_as_water(self) -> None:
        """暗いピクセルが水域として分類される。"""
        # Dark blueish pixels
        rgb = np.array([[[10, 20, 40]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert mask[0, 0]

    def test_bright_pixels_excluded(self) -> None:
        """明るいピクセルは水域から除外される。"""
        rgb = np.array([[[200, 190, 180]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]

    def test_dark_land_shadow_excluded(self) -> None:
        """暗い陸地の影（茶色系）は水域に含まれない。"""
        rgb = np.array([[[40, 35, 20]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]

    def test_dark_vegetation_excluded(self) -> None:
        """暗い植生（緑優勢）は水域に含まれない。"""
        rgb = np.array([[[15, 30, 15]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]

    def test_neutral_dark_classified_as_water(self) -> None:
        """中性的な暗いピクセル（ほぼ黒）は水域に含む。"""
        rgb = np.array([[[12, 14, 13]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert mask[0, 0]

    def test_bright_turquoise_coastal_water(self) -> None:
        """明るいターコイズ色の沿岸水域が水域として検出される。"""
        # Typical Sentinel-2 coastal turquoise: R<G, R<B, G≈B
        rgb = np.array([[[70, 140, 130]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert mask[0, 0]

    def test_bright_sediment_laden_water(self) -> None:
        """堆積物混じりの明るい水域が水域として検出される。"""
        # Yellowish-green shallow water near river mouths
        rgb = np.array([[[100, 150, 110]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert mask[0, 0]

    def test_mauve_offshore_water(self) -> None:
        """紫がかった沖合水域が水域として検出される。"""
        # Purple/mauve hue sometimes seen in offshore Sentinel-2
        rgb = np.array([[[70, 55, 80]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert mask[0, 0]

    def test_medium_blue_open_ocean(self) -> None:
        """中間的な明るさの外洋が水域として検出される。"""
        rgb = np.array([[[30, 40, 65]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert mask[0, 0]

    def test_bright_land_not_water(self) -> None:
        """明るい陸地（茶色や白い建物等）は水域に含めない。"""
        # Brown land
        rgb = np.array([[[160, 130, 90]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]

    def test_green_vegetation_not_water(self) -> None:
        """明るい植生が水域に含まれない。"""
        # Bright green vegetation: G >> R and G >> B
        rgb = np.array([[[50, 120, 40]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]

    def test_cloud_not_water(self) -> None:
        """白い雲が水域に含まれない。"""
        rgb = np.array([[[220, 225, 230]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]

    def test_urban_grey_not_water(self) -> None:
        """灰色の市街地が水域に含まれない。"""
        rgb = np.array([[[130, 130, 135]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]

    def test_red_soil_not_water(self) -> None:
        """赤い土壌が水域に含まれない。"""
        rgb = np.array([[[140, 80, 60]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]


class TestCoastlineIndex:
    def test_load_and_query(self, tmp_path: Path) -> None:
        """Shapefileを読み込み、バウンディングボックスでクエリできる。"""
        _write_test_shapefile(
            tmp_path / "lines.shp",
            [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]],
        )
        idx = CoastlineIndex(tmp_path / "lines.shp")
        # Query intersecting bbox
        results = idx.query((0.0, -0.5, 0.5, 0.5))
        assert len(results) >= 1

    def test_query_no_match(self, tmp_path: Path) -> None:
        """範囲外のクエリでは空リストが返る。"""
        _write_test_shapefile(
            tmp_path / "lines.shp",
            [[(0.0, 0.0), (1.0, 0.0)]],
        )
        idx = CoastlineIndex(tmp_path / "lines.shp")
        results = idx.query((10.0, 10.0, 11.0, 11.0))
        assert len(results) == 0

    def test_reuses_bbox_cache_on_second_init(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """2回目の初期化では bbox キャッシュを再利用する。"""
        shp_path = tmp_path / "lines.shp"
        _write_test_shapefile(
            shp_path,
            [[(0.0, 0.0), (1.0, 0.0)]],
        )

        CoastlineIndex(shp_path)
        assert water_mask_mod._bbox_cache_path(shp_path).exists()

        def _unexpected_read(_path: Path):
            raise AssertionError("bbox cache should have been reused")

        monkeypatch.setattr(water_mask_mod, "_read_shp_bboxes", _unexpected_read)

        idx = CoastlineIndex(shp_path)
        results = idx.query((0.0, -0.5, 0.5, 0.5))
        assert len(results) >= 1


class TestMakeWaterMaskFromCoastline:
    def test_no_coastlines_returns_all_water(self) -> None:
        """海岸線がないタイルは全域水域とみなす。"""
        rgb = np.full((10, 10, 3), 30, dtype=np.uint8)
        transform = _identity_transform()
        mask = make_water_mask_from_coastline([], rgb, transform, 10, 10)
        assert mask.all()

    def test_coastline_splits_tile(self) -> None:
        """海岸線によりタイルが水域と陸域に分割される。"""
        from rasterio.transform import from_bounds
        from shapely.geometry import LineString

        # Tile covers [0, 1] x [0, 1], coastline is a horizontal line at y=0.5
        transform = from_bounds(0.0, 0.0, 1.0, 1.0, 20, 20)
        coastline = LineString([(0.0, 0.5), (1.0, 0.5)])

        # Build RGB: top half (high y, low row) = dark blue water,
        # bottom half (low y, high row) = bright land.
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[:10, :] = [20, 30, 50]  # water (top rows = high y in geo)
        rgb[10:, :] = [180, 160, 140]  # land (bottom rows = low y in geo)

        mask = make_water_mask_from_coastline(
            [coastline], rgb, transform, 20, 20,
        )

        # Water region (top rows) should be True
        assert mask[:10, :].sum() > 0.7 * (10 * 20)
        # Land region (bottom rows) should be False
        assert mask[10:, :].sum() < 0.3 * (10 * 20)

    def test_coastline_mask_dtype(self) -> None:
        """戻り値がbool型のndarrayである。"""
        rgb = np.full((5, 5, 3), 30, dtype=np.uint8)
        transform = _identity_transform()
        mask = make_water_mask_from_coastline([], rgb, transform, 5, 5)
        assert mask.dtype == np.bool_

    def test_all_land_tile(self) -> None:
        """全域陸地のタイル（海岸線なし、明るいRGB）の場合、
        海岸線なしでも全域水域と返す（AND結合で既存マスクが絞り込む）。"""
        rgb = np.full((10, 10, 3), 200, dtype=np.uint8)
        transform = _identity_transform()
        mask = make_water_mask_from_coastline([], rgb, transform, 10, 10)
        # No coastlines → all-True (caller ANDs with RGB mask)
        assert mask.all()

    def test_vertical_coastline(self) -> None:
        """垂直な海岸線で左右に水域・陸域が分割される。"""
        from rasterio.transform import from_bounds
        from shapely.geometry import LineString

        transform = from_bounds(0.0, 0.0, 1.0, 1.0, 20, 20)
        coastline = LineString([(0.5, 0.0), (0.5, 1.0)])

        # Left half = water, right half = land
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[:, :10] = [20, 30, 50]  # water (left cols = low x)
        rgb[:, 10:] = [180, 160, 140]  # land (right cols = high x)

        mask = make_water_mask_from_coastline(
            [coastline], rgb, transform, 20, 20,
        )

        # Left side should be water
        assert mask[:, :10].sum() > 0.7 * (20 * 10)
        # Right side should be land
        assert mask[:, 10:].sum() < 0.3 * (20 * 10)


# ── Helpers ───────────────────────────────────────────────────────────────

from pathlib import Path


def _identity_transform():
    """Return a simple identity-like affine transform for testing."""
    from rasterio.transform import from_bounds
    return from_bounds(0.0, 0.0, 1.0, 1.0, 10, 10)


def _write_test_shapefile(
    path: Path,
    lines: list[list[tuple[float, float]]],
) -> None:
    """Write a minimal shapefile with polyline geometries for testing."""
    import shapefile as shp

    w = shp.Writer(str(path))
    w.field("id", "N")
    for i, coords in enumerate(lines):
        w.line([coords])
        w.record(i)
    w.close()
