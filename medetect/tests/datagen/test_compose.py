from __future__ import annotations

import math
import random

import numpy as np
import pytest

from medetect.datagen.compose import (
    _compose_one,
    _stamp_occupancy,
    blend_ship,
    compute_obb_corners,
    compute_ship_pixel_size,
    find_water_position,
    format_obb_label,
)


class TestComputeShipPixelSize:
    def test_destroyer_at_10m(self) -> None:
        """10 m/px 解像度での駆逐艦のピクセルサイズ。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "destroyer", lb_ratio=8.0, resolution_m=10.0, rng=rng,
        )
        # Destroyer ~150-190m → 15-19 px at 10m
        assert 10 <= length_px <= 25
        assert beam_px >= 2

    def test_fishing_trawler_at_2m(self) -> None:
        """2 m/px 解像度での漁船のピクセルサイズ。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "fishing_trawler", lb_ratio=5.0, resolution_m=2.0, rng=rng,
        )
        # Trawler ~15-40m → 7-20 px at 2m
        assert 5 <= length_px <= 30
        assert beam_px >= 2

    def test_minimum_pixel_size(self) -> None:
        """最小ピクセルサイズが保証される。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "fishing_trawler", lb_ratio=5.0, resolution_m=100.0, rng=rng,
        )
        assert beam_px >= 2
        assert length_px >= 3

    def test_unknown_class_uses_default(self) -> None:
        """未知のクラスでもデフォルトサイズで動作する。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "unknown_vessel", lb_ratio=6.0, resolution_m=5.0, rng=rng,
        )
        assert beam_px >= 2
        assert length_px >= 3

    def test_length_range_clamps_upper(self) -> None:
        """length_range の上限が適用される。"""
        rng = random.Random(0)
        results = [
            compute_ship_pixel_size(
                "destroyer", lb_ratio=8.0, resolution_m=1.0,
                rng=rng, length_range=(10.0, 50.0),
            )
            for _ in range(20)
        ]
        for _beam, length in results:
            assert length <= 52  # 50 m / 1 m/px + rounding tolerance

    def test_length_range_clamps_lower(self) -> None:
        """length_range の下限が適用される。"""
        rng = random.Random(0)
        results = [
            compute_ship_pixel_size(
                "fishing_trawler", lb_ratio=5.0, resolution_m=1.0,
                rng=rng, length_range=(80.0, 200.0),
            )
            for _ in range(20)
        ]
        for _beam, length in results:
            assert length >= 78  # 80 m / 1 m/px - rounding

    def test_length_range_none_uses_class_range(self) -> None:
        """length_range=None のとき制約なし（既存の動作を維持）。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "destroyer", lb_ratio=8.0, resolution_m=10.0, rng=rng, length_range=None,
        )
        assert length_px >= 3


class TestComputeObbCorners:
    def test_axis_aligned(self) -> None:
        """回転なしのOBBは軸揃いになる。"""
        corners = compute_obb_corners(50.0, 50.0, 10.0, 20.0, 0.0)
        assert len(corners) == 4
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        assert min(xs) == pytest.approx(45.0)
        assert max(xs) == pytest.approx(55.0)
        assert min(ys) == pytest.approx(40.0)
        assert max(ys) == pytest.approx(60.0)

    def test_rotated_90(self) -> None:
        """90度回転でOBBの幅と高さが入れ替わる。"""
        corners = compute_obb_corners(50.0, 50.0, 10.0, 20.0, math.pi / 2)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        assert min(xs) == pytest.approx(40.0, abs=0.1)
        assert max(xs) == pytest.approx(60.0, abs=0.1)
        assert min(ys) == pytest.approx(45.0, abs=0.1)
        assert max(ys) == pytest.approx(55.0, abs=0.1)

    def test_corners_form_rectangle(self) -> None:
        """4点が長方形を構成する（対角線の長さが等しい）。"""
        corners = compute_obb_corners(100.0, 100.0, 20.0, 40.0, 0.7)
        # Diagonals should be equal
        d1 = math.hypot(corners[2][0] - corners[0][0], corners[2][1] - corners[0][1])
        d2 = math.hypot(corners[3][0] - corners[1][0], corners[3][1] - corners[1][1])
        assert d1 == pytest.approx(d2, abs=0.01)


class TestFormatObbLabel:
    def test_format(self) -> None:
        """YOLO OBBラベル文字列のフォーマット。"""
        corners = [(10.0, 20.0), (30.0, 20.0), (30.0, 60.0), (10.0, 60.0)]
        label = format_obb_label(0, corners, img_w=100, img_h=100)
        parts = label.split()
        assert parts[0] == "0"
        assert len(parts) == 9  # class + 4 x,y pairs
        # Check normalization
        assert float(parts[1]) == pytest.approx(0.1)
        assert float(parts[2]) == pytest.approx(0.2)

    def test_coordinates_normalized(self) -> None:
        """座標が画像サイズで正規化される。"""
        corners = [(100.0, 200.0), (300.0, 200.0), (300.0, 400.0), (100.0, 400.0)]
        label = format_obb_label(1, corners, img_w=640, img_h=640)
        parts = label.split()
        for i in range(1, 9):
            val = float(parts[i])
            assert 0.0 <= val <= 1.0


class TestFindWaterPosition:
    def test_all_water_finds_position(self) -> None:
        """全面水域なら位置が見つかる。"""
        mask = np.ones((100, 100), dtype=bool)
        rng = random.Random(42)
        pos = find_water_position(mask, ship_w=10, ship_h=20, angle_rad=0.0, rng=rng)
        assert pos is not None
        cx, cy = pos
        assert 0 <= cx < 100
        assert 0 <= cy < 100

    def test_all_land_returns_none(self) -> None:
        """全面陸地ならNoneを返す。"""
        mask = np.zeros((100, 100), dtype=bool)
        rng = random.Random(42)
        pos = find_water_position(mask, ship_w=10, ship_h=20, angle_rad=0.0, rng=rng)
        assert pos is None

    def test_small_water_region_avoided(self) -> None:
        """小さな水域にはサイズの大きい船は置けない。"""
        mask = np.zeros((100, 100), dtype=bool)
        mask[48:52, 48:52] = True  # 4x4 water patch
        rng = random.Random(42)
        pos = find_water_position(mask, ship_w=20, ship_h=40, angle_rad=0.0, rng=rng)
        assert pos is None

    def test_occupied_area_avoided(self) -> None:
        """占有済みエリアには配置されない。"""
        mask = np.ones((100, 100), dtype=bool)
        # Occupy the entire mask except a corner
        mask[10:90, 10:90] = False
        rng = random.Random(42)
        pos = find_water_position(mask, ship_w=5, ship_h=5, angle_rad=0.0, rng=rng)
        # Should still find a position in the unoccupied strip
        assert pos is not None


class TestStampOccupancy:
    def test_marks_center_occupied(self) -> None:
        """船の中心が占有済みになる。"""
        occupancy = np.zeros((100, 100), dtype=bool)
        _stamp_occupancy(occupancy, cx=50, cy=50, w=10, h=20, angle_rad=0.0)
        assert occupancy[50, 50]

    def test_corners_outside_are_free(self) -> None:
        """小さい船をスタンプしても遠い角は未占有のまま。"""
        occupancy = np.zeros((100, 100), dtype=bool)
        _stamp_occupancy(occupancy, cx=50, cy=50, w=10, h=20, angle_rad=0.0)
        assert not occupancy[0, 0]

    def test_prevents_second_placement(self) -> None:
        """スタンプ後の占有マスクで同位置への再配置ができない。"""
        water = np.ones((100, 100), dtype=bool)
        occupancy = np.zeros((100, 100), dtype=bool)
        _stamp_occupancy(occupancy, cx=50, cy=50, w=30, h=60, angle_rad=0.0)
        available = water & ~occupancy
        rng = random.Random(42)
        # A large ship centered at 50,50 should no longer fit there
        pos = find_water_position(available, ship_w=30, ship_h=60, angle_rad=0.0, rng=rng)
        if pos is not None:
            cx, cy = pos
            # The found position must be away from the occupied center
            assert not (40 <= cx <= 60 and 30 <= cy <= 70)


class TestBlendShip:
    def test_modifies_background(self) -> None:
        """船をブレンドすると背景が変わる。"""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        ship = np.full((10, 5, 4), 200, dtype=np.uint8)  # Opaque white ship
        blend_ship(bg, ship, cx=50, cy=50, alpha_factor=1.0)
        assert bg[50, 50].sum() > 0

    def test_transparent_ship_no_change(self) -> None:
        """完全透明の船は背景を変えない。"""
        bg = np.full((100, 100, 3), 50, dtype=np.uint8)
        ship = np.zeros((10, 5, 4), dtype=np.uint8)  # Fully transparent
        original = bg.copy()
        blend_ship(bg, ship, cx=50, cy=50, alpha_factor=1.0)
        np.testing.assert_array_equal(bg, original)

    def test_clipping_at_boundary(self) -> None:
        """画像端でクリッピングされてエラーにならない。"""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        ship = np.full((20, 10, 4), 200, dtype=np.uint8)
        # Place at edge — should not raise
        blend_ship(bg, ship, cx=2, cy=2, alpha_factor=0.8)


class TestGeoScale:
    """geo_scale モードのテスト。"""

    @pytest.fixture()
    def tiny_tif(self, tmp_path: "pathlib.Path") -> "pathlib.Path":
        """低解像度(100 m/px)の小さな GeoTIFF を生成する。"""
        import pathlib

        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "bg.tif"
        size = 100  # pixels
        # Bounds span 100 * 100 m = 0.001° (roughly) — set explicitly in projected CRS
        transform = from_bounds(0, 0, 100 * size, 100 * size, size, size)
        data = np.full((3, size, size), 128, dtype=np.uint8)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(32654),  # UTM zone 54N — projected, ~100 m/px
            transform=transform,
        ) as dst:
            dst.write(data)
        return tif_path

    def test_geo_scale_1_produces_correct_size(
        self, tiny_tif: "pathlib.Path"
    ) -> None:
        """geo_scale=1.0 のとき出力タイルが image_size×image_size になる。"""
        rng = random.Random(0)
        result = _compose_one(
            tif_path=tiny_tif,
            svg_files=None,
            image_size=64,
            resolution=10.0,
            geo_scale=1.0,
            ships_per_image=(0, 0),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            ship_blur_sigma=0.5,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            rng=rng,
        )
        assert result is not None
        tile, _labels, _n_clusters = result
        assert tile.shape == (64, 64, 3)

    def test_geo_scale_none_uses_crs(self, tiny_tif: "pathlib.Path") -> None:
        """geo_scale=None のときは CRS ベースの解像度変換が行われる。"""
        rng = random.Random(1)
        result = _compose_one(
            tif_path=tiny_tif,
            svg_files=None,
            image_size=64,
            resolution=10.0,
            geo_scale=None,
            ships_per_image=(0, 0),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            ship_blur_sigma=0.5,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            rng=rng,
        )
        assert result is not None
        tile, _, _ = result
        assert tile.shape == (64, 64, 3)

    def test_geo_scale_05_upsamples(self, tiny_tif: "pathlib.Path") -> None:
        """geo_scale=0.5 のとき半分のTIFFピクセルを読みアップサンプルする。"""
        rng = random.Random(2)
        result = _compose_one(
            tif_path=tiny_tif,
            svg_files=None,
            image_size=64,
            resolution=10.0,
            geo_scale=0.5,
            ships_per_image=(0, 0),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            ship_blur_sigma=0.5,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            rng=rng,
        )
        assert result is not None
        tile, _, _ = result
        # src_tile = round(64 * 0.5) = 32, then resized to 64
        assert tile.shape == (64, 64, 3)


class TestShipSizeDistribution:
    """compute_ship_pixel_size の長さ分布が対数一様になっているか。"""

    def test_log_uniform_more_small_ships(self) -> None:
        """10-150m 範囲で生成すると中央値が (10+150)/2=80 より小さくなる。"""
        rng = random.Random(42)
        lengths = []
        for _ in range(2000):
            _bw, lh = compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng, length_range=(10.0, 150.0),
            )
            lengths.append(lh)  # at 1 m/px, length_px ≈ length_m
        median = sorted(lengths)[len(lengths) // 2]
        # Log-uniform median of [10, 150] = sqrt(10*150) ≈ 38.7
        # Linear uniform median would be ~80.
        assert median < 55, f"Median {median} too high — distribution not log-uniform"

    def test_log_uniform_still_produces_large(self) -> None:
        """大きな船もゼロではない。"""
        rng = random.Random(0)
        lengths = []
        for _ in range(500):
            _bw, lh = compute_ship_pixel_size(
                "carrier", 5.0, 1.0, rng, length_range=(10.0, 300.0),
            )
            lengths.append(lh)
        # carrier range (260-300) ∩ length_range (10-300) → 260-300m
        assert max(lengths) > 250, "No large ships generated"


class TestEdgeFeathering:
    """_render_ship で生成されるアルファ輪郭のフェザリング。"""

    def test_alpha_edges_are_soft(self) -> None:
        """生成された船のアルファチャンネル端部に中間値(0<a<255)が存在する。"""
        from medetect.datagen.compose import _render_ship

        rng = random.Random(7)
        # Generate a big enough ship so feathering is applied (min dim > 4)
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 5"'
            ' data-ship-class="destroyer" data-lb-ratio="5.0">'
            '<polygon points="0.5,0 0,5 1,5" fill="#888"/></svg>'
        )
        rgba, *_ = _render_ship(svg, 5.0, rng, 0.8, length_range=(80.0, 100.0))
        alpha = rgba[:, :, 3]
        # There should be pixels with partial transparency (feathered edges)
        partial = (alpha > 0) & (alpha < 255)
        assert partial.sum() > 0, "No feathered edge pixels found"
