from __future__ import annotations

import math
import pathlib
import random

import numpy as np
import pytest

from medetect.datagen.compose import (
    _SvgMeta,
    _blend_rgba_layer,
    _compose_one,
    _composite_rgba,
    _load_svg_metas,
    _natural_lb_ratio,
    _place_cluster,
    _ship_class_id,
    _stamp_occupancy,
    _svg_lb_weight,
    _write_dataset_yaml,
    blend_ship,
    compute_obb_corners,
    compute_ship_pixel_size,
    find_water_position,
    format_obb_label,
    is_dark_tile,
    make_nodata_mask,
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
            svg_metas=None,
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
            length_exponent=1.0,
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
            svg_metas=None,
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
            length_exponent=1.0,
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
            svg_metas=None,
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
            length_exponent=1.0,
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


class TestCompositeRgba:
    """_composite_rgba (Porter-Duff source-over) のテスト。"""

    def test_opaque_over_transparent(self) -> None:
        """透明背景に不透明船を重ねると船の色がそのまま出る。"""
        dst = np.zeros((20, 20, 4), dtype=np.uint8)
        src = np.full((10, 10, 4), 200, dtype=np.uint8)
        src[:, :, 3] = 255
        _composite_rgba(dst, src, 5, 5)
        assert dst[10, 10, 0] == 200
        assert dst[10, 10, 3] == 255

    def test_transparent_src_no_change(self) -> None:
        """透明なsrcを重ねてもdstは変わらない。"""
        dst = np.full((20, 20, 4), 100, dtype=np.uint8)
        src = np.zeros((10, 10, 4), dtype=np.uint8)
        _composite_rgba(dst, src, 5, 5)
        assert dst[10, 10, 0] == 100

    def test_two_ships_gap_shows_through(self) -> None:
        """並んだ船の隣接ピクセルの間に透明部分が残る。"""
        buf = np.zeros((50, 50, 4), dtype=np.uint8)
        # Ship A: columns 5-14
        ship_a = np.zeros((40, 10, 4), dtype=np.uint8)
        ship_a[:, :, :3] = 180
        ship_a[:, :, 3] = 255
        # Ship B: columns 16-25 (1px gap at column 15)
        ship_b = np.zeros((40, 10, 4), dtype=np.uint8)
        ship_b[:, :, :3] = 160
        ship_b[:, :, 3] = 255

        _composite_rgba(buf, ship_a, 5, 5)
        _composite_rgba(buf, ship_b, 16, 5)
        # The gap column (15) should remain transparent
        assert buf[20, 15, 3] == 0


class TestBlendRgbaLayer:
    """_blend_rgba_layer のテスト。"""

    def test_blends_with_alpha_factor(self) -> None:
        """アルファファクターが混合結果に影響する。"""
        bg = np.full((10, 10, 3), 100, dtype=np.uint8)
        layer = np.zeros((10, 10, 4), dtype=np.uint8)
        layer[3:7, 3:7, :3] = 200
        layer[3:7, 3:7, 3] = 255
        water_tint = np.array([40.0, 50.0, 60.0], dtype=np.float32)

        _blend_rgba_layer(bg, layer, 1.0, water_tint)
        # Interior pixel should be changed from 100
        assert bg[5, 5, 0] != 100
        # Pixel outside the layer alpha should remain 100
        assert bg[0, 0, 0] == 100


class TestAntiAliasedEdges:
    """スーパーサンプリング + PSF ブラーでエッジが滑らかになる確認。"""

    def test_alpha_edges_are_soft(self) -> None:
        """生成された船のアルファチャンネル端部に中間値(0<a<255)が存在する。"""
        from medetect.datagen.compose import _render_ship

        rng = random.Random(7)
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 5"'
            ' data-ship-class="destroyer" data-lb-ratio="5.0">'
            '<polygon points="0.5,0 0,5 1,5" fill="#888"/></svg>'
        )
        rgba, *_ = _render_ship(svg, 5.0, rng, 0.8, length_range=(80.0, 100.0))
        alpha = rgba[:, :, 3]
        # 4x supersample + PSF blur should produce anti-aliased edges
        partial = (alpha > 0) & (alpha < 255)
        assert partial.sum() > 0, "No anti-aliased edge pixels found"

    def test_interior_remains_opaque(self) -> None:
        """内部ピクセルが半透明にならない。"""
        from medetect.datagen.compose import _render_ship

        rng = random.Random(7)
        # Large ship so interior is well defined
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 5"'
            ' data-ship-class="destroyer" data-lb-ratio="5.0">'
            '<polygon points="0.5,0 0,5 1,5" fill="#888"/></svg>'
        )
        rgba, *_ = _render_ship(svg, 2.0, rng, 0.5, length_range=(80.0, 100.0))
        h, w = rgba.shape[:2]
        # Grab a central strip of the hull
        interior = rgba[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3, 3]
        if interior.size > 0:
            assert interior.min() > 200, (
                f"Interior alpha too low (min={interior.min()})"
            )


class TestLengthExponent:
    """length_exponent パラメータによるサイズ分布制御のテスト。"""

    def test_exponent_1_is_log_uniform(self) -> None:
        """exponent=1.0 は従来の対数一様分布と同等。"""
        rng = random.Random(42)
        lengths = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng,
                length_range=(10.0, 150.0), length_exponent=1.0,
            )[1]
            for _ in range(2000)
        ]
        median = sorted(lengths)[len(lengths) // 2]
        # Log-uniform median of [10, 150] ≈ sqrt(10*150) ≈ 38.7
        assert median < 55, f"Median {median} too high for log-uniform"

    def test_exponent_gt1_more_small(self) -> None:
        """exponent>1 にすると中央値が下がる（小さい船が増える）。"""
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        lengths_1 = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng1,
                length_range=(10.0, 150.0), length_exponent=1.0,
            )[1]
            for _ in range(2000)
        ]
        lengths_3 = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng2,
                length_range=(10.0, 150.0), length_exponent=3.0,
            )[1]
            for _ in range(2000)
        ]
        median_1 = sorted(lengths_1)[len(lengths_1) // 2]
        median_3 = sorted(lengths_3)[len(lengths_3) // 2]
        assert median_3 < median_1, (
            f"exponent=3 median ({median_3}) should be < exponent=1 ({median_1})"
        )

    def test_exponent_lt1_more_large(self) -> None:
        """exponent<1 にすると中央値が上がる（大きい船が増える）。"""
        rng1 = random.Random(7)
        rng2 = random.Random(7)
        lengths_1 = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng1,
                length_range=(10.0, 150.0), length_exponent=1.0,
            )[1]
            for _ in range(2000)
        ]
        lengths_05 = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng2,
                length_range=(10.0, 150.0), length_exponent=0.3,
            )[1]
            for _ in range(2000)
        ]
        median_1 = sorted(lengths_1)[len(lengths_1) // 2]
        median_05 = sorted(lengths_05)[len(lengths_05) // 2]
        assert median_05 > median_1, (
            f"exponent=0.3 median ({median_05}) should be > exponent=1 ({median_1})"
        )


class TestLandOnlyNegativeExample:
    """陸地のみのタイルがネガティブサンプル（船なし）として出力されるテスト。"""

    @pytest.fixture()
    def land_only_tif(self, tmp_path: "pathlib.Path") -> "pathlib.Path":
        """水域なし（全面陸地相当）の GeoTIFF を生成する。"""
        import pathlib

        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "land_visual.tif"
        size = 200
        transform = from_bounds(0, 0, 100 * size, 100 * size, size, size)
        # Bright brownish land — clearly not water
        data = np.full((3, size, size), 0, dtype=np.uint8)
        data[0, :, :] = 120  # R
        data[1, :, :] = 100  # G
        data[2, :, :] = 70   # B
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(32654),
            transform=transform,
        ) as dst:
            dst.write(data)
        return tif_path

    def test_land_tile_returns_negative_example(
        self, land_only_tif: "pathlib.Path"
    ) -> None:
        """水域なしタイルが船なし（ラベル空）で返る。"""
        rng = random.Random(42)
        result = _compose_one(
            tif_path=land_only_tif,
            svg_metas=None,
            image_size=64,
            resolution=10.0,
            geo_scale=1.0,
            ships_per_image=(3, 5),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.3,
            ship_blur_sigma=0.5,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            rng=rng,
        )
        assert result is not None, "Land-only tile should return (tile, [], 0), not None"
        tile, labels, n_clusters = result
        assert tile.shape == (64, 64, 3)
        assert labels == []
        assert n_clusters == 0


class TestOverlapPrevention:
    """船の重なり防止のテスト。"""

    def test_find_water_position_avoids_occupied(self) -> None:
        """占有マスクが考慮されて既存船との重複を避ける。"""
        water = np.ones((200, 200), dtype=bool)
        occupancy = np.zeros((200, 200), dtype=bool)
        # Occupy a large area in the center
        _stamp_occupancy(occupancy, cx=100, cy=100, w=60, h=120, angle_rad=0.0)
        available = water & ~occupancy
        rng = random.Random(0)
        positions = []
        for _ in range(50):
            pos = find_water_position(
                available, ship_w=10, ship_h=20, angle_rad=0.0, rng=rng,
            )
            if pos is not None:
                positions.append(pos)
        # All found positions should be outside the occupied area
        for cx, cy in positions:
            assert not (70 <= cx <= 130 and 40 <= cy <= 160), (
                f"Ship placed at ({cx}, {cy}) inside occupied zone"
            )


class TestNaturalLbRatio:
    """_natural_lb_ratio の小型船・大型船の物理的妥当性検証。"""

    def test_small_ship_low_lb(self) -> None:
        """5mディンギー等、小型船は低いlb_ratio。"""
        assert _natural_lb_ratio(5.0) < 4.0

    def test_large_ship_higher_lb(self) -> None:
        """200m驱逐艦等、大型船は高いlb_ratio。"""
        assert _natural_lb_ratio(200.0) > _natural_lb_ratio(20.0)

    def test_capped_at_10(self) -> None:
        """10が上限。"""
        assert _natural_lb_ratio(10000.0) == 10.0


class TestSvgLbWeight:
    """_svg_lb_weight の重み計算の検証。"""

    def test_natural_lb_gets_full_weight(self) -> None:
        """自然なlb_ratioの船は重み1.0。"""
        lb = _natural_lb_ratio(15.0)  # natural lb at 15 m
        assert _svg_lb_weight(lb, 15.0) == 1.0

    def test_excess_lb_gets_lower_weight(self) -> None:
        """lb_ratioが自然値の1.5倍を超えると重みが下がる。"""
        w_bad = _svg_lb_weight(12.0, 10.0)   # very high lb for a 10 m target
        w_good = _svg_lb_weight(3.5, 10.0)   # low lb, appropriate for 10 m
        assert w_bad < w_good

    def test_weight_positive(self) -> None:
        """重みは常に正。"""
        assert _svg_lb_weight(15.0, 5.0) > 0.0


class TestLoadSvgMetas:
    """_load_svg_metas のメタデータ読み込みまとめの検証。"""

    def test_reads_lb_ratio(self, tmp_path: "pathlib.Path") -> None:
        """SVGファイルからlb_ratioを正しく読み取る。"""
        import pathlib

        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4.5"'
            ' data-ship-class="fishing_trawler" data-lb-ratio="4.5">'
            '<polygon points="0.5,0 0,4.5 1,4.5" fill="#666"/></svg>'
        )
        svg_path = tmp_path / "test_ship.svg"
        svg_path.write_text(svg_content, encoding="utf-8")

        metas = _load_svg_metas([svg_path])
        assert len(metas) == 1
        assert isinstance(metas[0], _SvgMeta)
        assert metas[0].path == svg_path
        assert metas[0].lb_ratio == 4.5


class TestIsDarkTile:
    """衛星画像の帯状真っ黒領域 (blackout tile) 検出のテスト。"""

    def test_all_black_is_dark(self) -> None:
        """全黒タイルは暗いと判定される。"""
        tile = np.zeros((64, 64, 3), dtype=np.uint8)
        assert is_dark_tile(tile)

    def test_normal_image_is_not_dark(self) -> None:
        """通常の輝度を持つタイルは暗いと判定されない。"""
        tile = np.full((64, 64, 3), 80, dtype=np.uint8)
        assert not is_dark_tile(tile)

    def test_custom_threshold(self) -> None:
        """カスタム閾値が適用される。"""
        tile = np.full((64, 64, 3), 5, dtype=np.uint8)  # mean=5
        assert not is_dark_tile(tile, threshold=5.0)   # mean == threshold → not dark
        assert is_dark_tile(tile, threshold=6.0)        # mean < threshold → dark

    def test_stripe_scenario(self) -> None:
        """帯状に真っ黒な領域が含まれるタイルは暗いと判定される。"""
        # 半分が黒、半分が通常輝度のタイル → mean ≈ 40 → 暗くない
        tile = np.full((64, 64, 3), 80, dtype=np.uint8)
        tile[:32, :, :] = 0  # 上半分が黒いストライプ
        assert not is_dark_tile(tile)  # mean ≈ 40、閾値 10 より大きい

        # ほぼ全体が黒で僅かに輝度があるタイル → dark
        mostly_dark = np.zeros((64, 64, 3), dtype=np.uint8)
        mostly_dark[60:64, 60:64, :] = 80  # ごく一部だけ明るい
        assert is_dark_tile(mostly_dark)


class TestMakeNodataMask:
    """純黒 (#000000) の no-data 領域検出のテスト。"""

    def test_all_black_is_nodata(self) -> None:
        """全黒タイルは全ピクセルが no-data とマークされる。"""
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = make_nodata_mask(tile)
        assert mask.all()

    def test_normal_tile_has_no_nodata(self) -> None:
        """通常の輝度を持つタイルは no-data ピクセルを含まない。"""
        tile = np.full((4, 4, 3), 80, dtype=np.uint8)
        mask = make_nodata_mask(tile)
        assert not mask.any()

    def test_partial_black_strip(self) -> None:
        """一部が黒いストライプを含むタイルは黒部分のみ no-data。"""
        tile = np.full((8, 8, 3), 100, dtype=np.uint8)
        tile[:4, :, :] = 0  # 上半分が黒
        mask = make_nodata_mask(tile)
        assert mask[:4, :].all()
        assert not mask[4:, :].any()

    def test_single_channel_zero_is_not_nodata(self) -> None:
        """1チャンネルだけ 0 でも no-data ではない（#000000 = 全チャンネル 0 のみ）。"""
        tile = np.full((4, 4, 3), 0, dtype=np.uint8)
        tile[:, :, 0] = 50  # R だけ非ゼロ
        mask = make_nodata_mask(tile)
        assert not mask.any()

    def test_nodata_excluded_from_water_mask(self) -> None:
        """pure black (#000000) ピクセルは RGB ウォーターマスクから除外される。"""
        from medetect.datagen.water_mask import make_water_mask_from_rgb

        # 暗い水域と同じ輝度特性を持つが、黒いため no-data 扱いになるタイル
        tile = np.zeros((8, 8, 3), dtype=np.uint8)
        tile[4:, :, :] = 30  # 下半分は暗い水域ふう

        water = make_water_mask_from_rgb(tile)
        nodata = make_nodata_mask(tile)
        water_clean = water & ~nodata

        # 上半分 (pure black) は水マスクから除外されている
        assert not water_clean[:4, :].any()
        # 下半分は引き続き水として検出される
        assert water_clean[4:, :].any()


class TestNaturalLbRatio:
    """_natural_lb_ratio の物理的妥当性検証。"""

    def test_small_ship_low_lb(self) -> None:
        """5m ディンギー等、小型船は低い lb_ratio。"""
        assert _natural_lb_ratio(5.0) < 4.0

    def test_large_ship_higher_lb(self) -> None:
        """200m 駆逐艦等、大型船は高い lb_ratio。"""
        assert _natural_lb_ratio(200.0) > _natural_lb_ratio(20.0)

    def test_capped_at_10(self) -> None:
        """10 が上限。"""
        assert _natural_lb_ratio(10000.0) == 10.0

    def test_monotone_increasing(self) -> None:
        """lb_ratio は船の長さに対して単調増加する。"""
        lengths = [5.0, 20.0, 50.0, 100.0, 200.0, 300.0]
        values = [_natural_lb_ratio(l) for l in lengths]
        for a, b in zip(values, values[1:]):
            assert a <= b


class TestSvgLbWeight:
    """_svg_lb_weight の重み計算の検証。"""

    def test_natural_lb_gets_full_weight(self) -> None:
        """自然な lb_ratio の船は重み 1.0。"""
        lb = _natural_lb_ratio(15.0)
        assert _svg_lb_weight(lb, 15.0) == 1.0

    def test_excess_lb_gets_lower_weight(self) -> None:
        """lb_ratio が自然値の 1.5 倍を超えると重みが下がる。"""
        w_bad = _svg_lb_weight(12.0, 10.0)   # 10m 目標に対し過大な lb
        w_good = _svg_lb_weight(3.5, 10.0)   # 10m 目標に適した lb
        assert w_bad < w_good

    def test_weight_always_positive(self) -> None:
        """重みは常に正。"""
        assert _svg_lb_weight(15.0, 5.0) > 0.0

    def test_small_target_prefers_low_lb(self) -> None:
        """小型船ターゲットでは、低 lb_ratio の SVG が高く評価される。"""
        w_stubby = _svg_lb_weight(4.0, 10.0)   # trawler-like lb
        w_slender = _svg_lb_weight(9.0, 10.0)  # destroyer-like lb
        assert w_stubby > w_slender


class TestLoadSvgMetas:
    """_load_svg_metas のメタデータ読み込みまとめの検証。"""

    def test_reads_lb_ratio(self, tmp_path) -> None:
        """SVG ファイルから lb_ratio を正しく読み取る。"""
        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4.5"'
            ' data-ship-class="fishing_trawler" data-lb-ratio="4.5">'
            '<polygon points="0.5,0 0,4.5 1,4.5" fill="#666"/></svg>'
        )
        svg_path = tmp_path / "test_ship.svg"
        svg_path.write_text(svg_content, encoding="utf-8")

        metas = _load_svg_metas([svg_path])
        assert len(metas) == 1
        assert isinstance(metas[0], _SvgMeta)
        assert metas[0].path == svg_path
        assert metas[0].lb_ratio == 4.5

    def test_multiple_files(self, tmp_path) -> None:
        """複数ファイルのメタが順番通り返る。"""
        lb_values = [3.8, 6.5, 9.0]
        paths = []
        for i, lb in enumerate(lb_values):
            content = (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 {lb}"'
                f' data-ship-class="patrol" data-lb-ratio="{lb}">'
                f'<polygon points="0.5,0 0,{lb} 1,{lb}" fill="#888"/></svg>'
            )
            p = tmp_path / f"ship_{i}.svg"
            p.write_text(content, encoding="utf-8")
            paths.append(p)

        metas = _load_svg_metas(paths)
        assert [m.lb_ratio for m in metas] == lb_values


class TestShipClassId:
    """_ship_class_id による大小クラス判定の検証。"""

    def test_no_threshold_returns_base_id(self) -> None:
        """しきい値なしのとき、常に base class_id を返す。"""
        assert _ship_class_id(100, 10.0, 0, None) == 0

    def test_below_threshold_returns_small(self) -> None:
        """長さがしきい値未満なら small (class_id) を返す。"""
        # 5 px * 10 m/px = 50 m < 100 m threshold
        assert _ship_class_id(5, 10.0, 0, 100.0) == 0

    def test_at_threshold_returns_large(self) -> None:
        """長さがしきい値ちょうどなら large (class_id + 1) を返す。"""
        # 10 px * 10 m/px = 100 m == 100 m threshold
        assert _ship_class_id(10, 10.0, 0, 100.0) == 1

    def test_above_threshold_returns_large(self) -> None:
        """長さがしきい値超なら large (class_id + 1) を返す。"""
        # 15 px * 10 m/px = 150 m > 100 m threshold
        assert _ship_class_id(15, 10.0, 0, 100.0) == 1

    def test_custom_base_class_id(self) -> None:
        """base class_id が 0 以外でも正しく動作する。"""
        assert _ship_class_id(5, 10.0, 2, 100.0) == 2   # small
        assert _ship_class_id(15, 10.0, 2, 100.0) == 3   # large


class TestWriteDatasetYaml:
    """_write_dataset_yaml の出力検証。"""

    def test_single_class_without_threshold(self, tmp_path) -> None:
        """しきい値なしのとき、単一クラス ship を出力する。"""
        _write_dataset_yaml(tmp_path, 0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  0: ship\n" in content
        assert "ship_small" not in content
        assert "ship_large" not in content

    def test_two_classes_with_threshold(self, tmp_path) -> None:
        """しきい値ありのとき、ship_small と ship_large の2クラスを出力する。"""
        _write_dataset_yaml(tmp_path, 0, size_threshold=100.0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  0: ship_small\n" in content
        assert "  1: ship_large\n" in content

    def test_params_written_as_comments(self, tmp_path) -> None:
        """生成パラメータがコメントとして書き込まれる。"""
        params = {"count": 100, "resolution": 10.0, "size_threshold": 80.0}
        _write_dataset_yaml(tmp_path, 0, size_threshold=80.0, params=params)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "# Generation parameters:" in content
        assert "#   count: 100" in content
        assert "#   resolution: 10.0" in content
        assert "#   size_threshold: 80.0" in content

    def test_no_params_no_comment(self, tmp_path) -> None:
        """パラメータなしのとき、コメント行がない。"""
        _write_dataset_yaml(tmp_path, 0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "# Generation parameters:" not in content

    def test_custom_class_id_with_threshold(self, tmp_path) -> None:
        """class_id が 0 以外のとき、正しい ID で出力される。"""
        _write_dataset_yaml(tmp_path, 3, size_threshold=50.0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  3: ship_small\n" in content
        assert "  4: ship_large\n" in content


class TestAugmentTile:
    """augment_tile によるタイルの色オーグメンテーション検証。"""

    def test_output_shape_unchanged(self) -> None:
        """出力タイルの形状が変わらない。"""
        from medetect.datagen.compose import augment_tile

        tile = np.full((64, 64, 3), 100, dtype=np.uint8)
        rng = random.Random(0)
        result = augment_tile(tile, rng)
        assert result.shape == tile.shape
        assert result.dtype == np.uint8

    def test_output_differs_from_input(self) -> None:
        """オーグメンテーション後は元と異なる値になる（特定のseedで）。"""
        from medetect.datagen.compose import augment_tile

        tile = np.full((64, 64, 3), 100, dtype=np.uint8)
        rng = random.Random(42)
        result = augment_tile(tile, rng)
        assert not np.array_equal(result, tile)

    def test_different_seeds_produce_different_results(self) -> None:
        """異なるシードで異なるオーグメンテーション結果になる。"""
        from medetect.datagen.compose import augment_tile

        tile = np.full((64, 64, 3), 80, dtype=np.uint8)
        results = []
        for seed in range(10):
            rng = random.Random(seed)
            results.append(augment_tile(tile, rng).mean())
        # At least some seeds should produce different means
        unique_means = set(round(m, 1) for m in results)
        assert len(unique_means) >= 3, f"Too few unique augmentations: {unique_means}"

    def test_values_clipped_to_uint8(self) -> None:
        """オーグメンテーション後の値が0-255の範囲に収まる。"""
        from medetect.datagen.compose import augment_tile

        # Very bright and very dark tiles
        for val in [0, 5, 250, 255]:
            tile = np.full((32, 32, 3), val, dtype=np.uint8)
            for seed in range(5):
                rng = random.Random(seed)
                result = augment_tile(tile, rng)
                assert result.min() >= 0
                assert result.max() <= 255

    def test_channels_shifted_independently(self) -> None:
        """チャンネル別のゲインが独立に適用される。"""
        from medetect.datagen.compose import augment_tile

        tile = np.full((64, 64, 3), 100, dtype=np.uint8)
        # Run many seeds and check that channels diverge at least sometimes
        channel_diffs = 0
        for seed in range(20):
            rng = random.Random(seed)
            result = augment_tile(tile, rng)
            means = [result[:, :, c].mean() for c in range(3)]
            if max(means) - min(means) > 1:
                channel_diffs += 1
        assert channel_diffs > 0, "Channels never differ"

    def test_typical_water_tile_augmented(self) -> None:
        """典型的な暗い海面タイルに色の多様性が出る。"""
        from medetect.datagen.compose import augment_tile

        # Dark ocean-like tile
        tile = np.zeros((64, 64, 3), dtype=np.uint8)
        tile[:, :, 0] = 15  # R
        tile[:, :, 1] = 25  # G
        tile[:, :, 2] = 40  # B
        means = []
        for seed in range(20):
            rng = random.Random(seed)
            result = augment_tile(tile, rng)
            means.append(tuple(result.mean(axis=(0, 1))))
        # Check there is visible diversity: blue channel mean should vary
        blue_means = [m[2] for m in means]
        assert max(blue_means) - min(blue_means) > 5, (
            f"Insufficient diversity: blue means range = {max(blue_means) - min(blue_means):.1f}"
        )


# ── OBB area helper ──────────────────────────────────────────────────────


def _obb_area(label: str, img_size: int = 200) -> float:
    """YOLO OBB ラベル文字列からピクセル面積を計算する（Shoelace 公式）。"""
    parts = label.split()
    coords = [float(v) * img_size for v in parts[1:]]
    x1, y1, x2, y2, x3, y3, x4, y4 = coords
    return 0.5 * abs(
        (x1 * y2 - x2 * y1)
        + (x2 * y3 - x3 * y2)
        + (x3 * y4 - x4 * y3)
        + (x4 * y1 - x1 * y4)
    )


class TestPlaceCluster:
    """_place_cluster の均一モード / 混合モードの動作検証。"""

    _IMAGE_SIZE = 200

    @pytest.fixture()
    def scene(self):
        """全面水域の 200×200 シーン。"""
        size = self._IMAGE_SIZE
        return {
            "water_mask": np.ones((size, size), dtype=bool),
            "occupancy": np.zeros((size, size), dtype=bool),
            "background": np.full((size, size, 3), 60, dtype=np.uint8),
        }

    def test_uniform_cluster_produces_labels(self, scene) -> None:
        """均一クラスター (mixed_prob=0) がラベルを生成する。"""
        rng = random.Random(42)
        labels = _place_cluster(
            scene["water_mask"], scene["occupancy"], None,
            resolution_m=10.0, rng=rng,
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"],
            length_range=(50.0, 80.0),
            mixed_prob=0.0,
        )
        assert len(labels) > 0

    def test_mixed_cluster_produces_labels(self, scene) -> None:
        """混合クラスター (mixed_prob=1) がラベルを生成する。"""
        rng = random.Random(42)
        labels = _place_cluster(
            scene["water_mask"], scene["occupancy"], None,
            resolution_m=10.0, rng=rng,
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"],
            length_range=(50.0, 80.0),
            mixed_prob=1.0,
        )
        assert len(labels) > 0

    def test_uniform_cluster_ships_similar_size(self, scene) -> None:
        """均一クラスターでは各船の OBB 面積が元の ±21% 以内に収まる。"""
        rng = random.Random(7)
        labels = _place_cluster(
            scene["water_mask"], scene["occupancy"], None,
            resolution_m=10.0, rng=rng,
            cluster_size_range=(3, 3),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"],
            length_range=(70.0, 90.0),
            mixed_prob=0.0,
        )
        assert len(labels) >= 2, "期待通りのラベル数が得られなかった"
        areas = [_obb_area(l, self._IMAGE_SIZE) for l in labels]
        # Ships at i>0 are rendered at ±10% of the reference size,
        # giving a maximum area ratio of (1.1/0.9)^2 ≈ 1.49.
        # We tolerate 2× to account for the first ship (rendered independently).
        ratio = max(areas) / min(areas)
        assert ratio < 2.0, (
            f"Uniform cluster ships too different in size (ratio={ratio:.2f})"
        )

    def test_mixed_cluster_shows_size_variety_over_runs(self, scene) -> None:
        """混合クラスターを広い length_range で繰り返すと OBB 面積に広い分散が出る。"""
        all_areas: list[float] = []
        for seed in range(15):
            wm = scene["water_mask"].copy()
            oc = np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool)
            rng = random.Random(seed)
            labels = _place_cluster(
                wm, oc, None,
                resolution_m=5.0, rng=rng,
                cluster_size_range=(3, 3),
                blur_sigma=0.0,
                alpha_range=(0.8, 0.9),
                class_id=0,
                image_size=self._IMAGE_SIZE,
                background=scene["background"].copy(),
                length_range=None,  # 全クラスの範囲 → サイズ分散が大きい
                mixed_prob=1.0,
            )
            all_areas.extend(_obb_area(l, self._IMAGE_SIZE) for l in labels)
        assert len(all_areas) >= 4, "テスト成立に必要なラベル数を得られなかった"
        ratio = max(all_areas) / min(all_areas)
        assert ratio > 2.0, (
            f"Mixed clusters should exhibit size diversity (ratio={ratio:.2f})"
        )
