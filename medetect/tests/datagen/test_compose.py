from __future__ import annotations

import math
import pathlib
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

import medetect.datagen.compose as compose_mod

from medetect.datagen.compose import (
    _SvgMeta,
    _blend_rgba_layer,
    _compose_one,
    _composite_rgba,
    _false_source_grid,
    _geometry_projection_extents,
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
    generate_false_negatives,
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

    def test_blends_without_water_tint(self) -> None:
        """water_tint が None の場合、船の色をそのまま使う。"""
        bg = np.full((10, 10, 3), 100, dtype=np.uint8)
        layer = np.zeros((10, 10, 4), dtype=np.uint8)
        layer[3:7, 3:7, :3] = 200
        layer[3:7, 3:7, 3] = 255

        _blend_rgba_layer(bg, layer, 1.0, None)
        # Interior pixel should be blended without tinting
        assert bg[5, 5, 0] != 100
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

    def test_hard_reject_above_twice_natural(self) -> None:
        """natural の 2.0 倍を超える lb_ratio は hard-reject (weight=0.0)。"""
        # 5m 船: natural≈3.15, 2×natural≈6.30 → lb=15.0 は超過
        assert _svg_lb_weight(15.0, 5.0) == 0.0

    def test_within_twice_natural_has_positive_weight(self) -> None:
        """natural の 2.0 倍以内の lb_ratio は正の重みを返す。"""
        # 5m 船: natural≈3.15, 2×natural≈6.30 → lb=6.0 は OK
        assert _svg_lb_weight(6.0, 5.0) > 0.0


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

    def test_hard_reject_above_twice_natural(self) -> None:
        """natural の 2.0 倍を超える lb_ratio は hard-reject (weight=0.0)。"""
        # 5m 船: natural≈3.15, 2×natural≈6.30 → lb=15.0 は超過
        assert _svg_lb_weight(15.0, 5.0) == 0.0

    def test_within_twice_natural_has_positive_weight(self) -> None:
        """natural の 2.0 倍以内の lb_ratio は正の重みを返す。"""
        # 5m 船: natural≈3.15, 2×natural≈6.30 → lb=6.0 は OK
        assert _svg_lb_weight(6.0, 5.0) > 0.0

    def test_hard_reject_boundary_small_ship(self) -> None:
        """小型船 (15m) で、駆逐艦相当の高 lb は 0.0 になる。"""
        # 15m 船: natural=3.45, 2×natural=6.90
        # patrol SVG の lb=5.5〜10.0 → 上端 (10.0) はリジェクト
        assert _svg_lb_weight(10.0, 15.0) == 0.0
        # lb=6.0 は 2×natural=6.90 以内 → OK
        assert _svg_lb_weight(6.0, 15.0) > 0.0

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


class TestGenerateDatasetParams:
    """generate_dataset の記録パラメータ整合を検証する。"""

    def test_removed_debug_params_are_not_written(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """削除対象のデバッグ系パラメータは dataset.yaml 用 params に含めない。"""
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        (bg_dir / "scene_visual.tif").write_bytes(b"placeholder")

        captured: dict[str, object] = {}

        class _DummyExecutor:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def submit(self, *args, **kwargs):
                msg = "count=0 should not submit compose tasks"
                raise AssertionError(msg)

        def _capture_yaml(
            output_dir: pathlib.Path,
            class_id: int,
            *,
            size_threshold: float | None = None,
            params: dict[str, object] | None = None,
        ) -> None:
            captured["params"] = dict(params or {})

        monkeypatch.setattr(compose_mod.concurrent.futures, "ProcessPoolExecutor", _DummyExecutor)
        monkeypatch.setattr(compose_mod, "_write_dataset_yaml", _capture_yaml)

        compose_mod.generate_dataset(
            bg_dir=bg_dir,
            output_dir=tmp_path / "out",
            count=0,
            max_workers=1,
        )

        params = captured["params"]
        assert isinstance(params, dict)
        assert "force_tight_clusters" not in params
        assert "debug_bg_color" not in params
        assert "disable_water_tint" not in params


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


def _parse_obb_polygon(label: str, img_size: int) -> list[tuple[float, float]]:
    """YOLO OBB ラベルから4頂点のポリゴンを返す。"""
    parts = label.split()
    coords = [float(v) * img_size for v in parts[1:]]
    return [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]


def _cross_2d(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    n = len(poly)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def _segment_intersect(
    p1: tuple[float, float], p2: tuple[float, float],
    p3: tuple[float, float], p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Two segments intersection point (or None)."""
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p4[0] - p3[0], p4[1] - p3[1])
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0] + t * d1[0], p1[1] + t * d1[1])
    return None


def _point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    n = len(poly)
    inside = False
    x, y = pt
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_intersection(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman polygon clipping: clip poly_a by poly_b."""
    output = list(poly_a)
    n = len(poly_b)
    for i in range(n):
        if not output:
            return []
        edge_start = poly_b[i]
        edge_end = poly_b[(i + 1) % n]
        inp = output
        output = []
        for j in range(len(inp)):
            curr = inp[j]
            prev = inp[j - 1]
            curr_inside = _cross_2d(edge_start, edge_end, curr) >= 0
            prev_inside = _cross_2d(edge_start, edge_end, prev) >= 0
            if curr_inside:
                if not prev_inside:
                    ix = _segment_intersect(prev, curr, edge_start, edge_end)
                    if ix:
                        output.append(ix)
                output.append(curr)
            elif prev_inside:
                ix = _segment_intersect(prev, curr, edge_start, edge_end)
                if ix:
                    output.append(ix)
    return output


def _polygon_iou(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> float:
    """Compute IoU between two convex polygons."""
    inter = _polygon_intersection(poly_a, poly_b)
    if len(inter) < 3:
        return 0.0
    inter_area = _polygon_area(inter)
    area_a = _polygon_area(poly_a)
    area_b = _polygon_area(poly_b)
    union = area_a + area_b - inter_area
    if union < 1e-12:
        return 0.0
    return inter_area / union


def _point_to_segment_dist(
    pt: tuple[float, float],
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> float:
    """Minimum distance from a point to a line segment."""
    dx, dy = seg_b[0] - seg_a[0], seg_b[1] - seg_a[1]
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(pt[0] - seg_a[0], pt[1] - seg_a[1])
    t = max(0.0, min(1.0, ((pt[0] - seg_a[0]) * dx + (pt[1] - seg_a[1]) * dy) / len_sq))
    proj_x = seg_a[0] + t * dx
    proj_y = seg_a[1] + t * dy
    return math.hypot(pt[0] - proj_x, pt[1] - proj_y)


def _polygon_min_distance(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> float:
    """Minimum distance between two convex polygons (0 if overlapping)."""
    # Check overlap
    inter = _polygon_intersection(poly_a, poly_b)
    if len(inter) >= 3 and _polygon_area(inter) > 1e-6:
        return 0.0
    # Min vertex-to-edge distance
    best = float("inf")
    for poly_x, poly_y in [(poly_a, poly_b), (poly_b, poly_a)]:
        n = len(poly_y)
        for pt in poly_x:
            for i in range(n):
                d = _point_to_segment_dist(pt, poly_y[i], poly_y[(i + 1) % n])
                if d < best:
                    best = d
    return best


def _count_connected_components(mask: np.ndarray, min_size: int = 1) -> int:
    """Count 8-connected components in a boolean mask."""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            count = 0
            while stack:
                cx, cy = stack.pop()
                count += 1
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if count >= min_size:
                components += 1
    return components


class _ForcedLayoutRandom(random.Random):
    """Random subclass that forces a specific cluster layout."""

    def __init__(self, seed: int, layout: str, base_angle: float = 0.0) -> None:
        super().__init__(seed)
        self._layout = layout
        self._base_angle = base_angle
        self._uniform_calls = 0

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        pop_list = list(population)
        _layout_map = {
            "flush": "raft_tight",
            "partial": "raft_open",
            "gapped": "area_scattered",
        }
        if set(pop_list) == {"raft_tight", "raft_open", "area_scattered"} and k == 1:
            return [_layout_map.get(self._layout, self._layout)]
        return super().choices(
            population, weights=weights, cum_weights=cum_weights, k=k,
        )

    def uniform(self, a, b):
        if self._uniform_calls == 0 and a == 0 and b == 360:
            self._uniform_calls += 1
            return self._base_angle
        if a < 0 < b and max(abs(a), abs(b)) <= 2.0:
            self._uniform_calls += 1
            return 0.0
        self._uniform_calls += 1
        return super().uniform(a, b)


def _make_tapered_hull_rgba(beam_px: int, length_px: int) -> np.ndarray:
    """Create a simple tapered hull mask for tight-placement tests."""
    img = Image.new("RGBA", (beam_px, length_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    tip = max(1, round(length_px * 0.16))
    half_x = (beam_px - 1) / 2.0
    points = [
        (half_x, 0),
        (beam_px - 1, tip),
        (beam_px - 1, max(tip + 1, length_px - tip - 1)),
        (half_x, length_px - 1),
        (0, max(tip + 1, length_px - tip - 1)),
        (0, tip),
    ]
    draw.polygon(points, fill=(220, 220, 220, 255))
    return np.array(img, dtype=np.uint8)


def _resolve_ship_dimensions_sequence_factory(
    sizes: list[tuple[int, int]],
):
    """Return a deterministic _resolve_ship_dimensions mock with predefined sizes."""
    calls = {"count": 0}

    def _mock_resolve_ship_dimensions(
        svg_text: str,
        resolution_m: float,
        rng: random.Random,
        length_range: tuple[float, float] | None = None,
        length_exponent: float = 1.0,
    ) -> tuple[str, int, int, float]:
        index = min(calls["count"], len(sizes) - 1)
        beam_px, length_px = sizes[index]
        calls["count"] += 1
        lb_ratio = length_px / max(beam_px, 1)
        return "mock_hull", beam_px, length_px, lb_ratio

    return _mock_resolve_ship_dimensions


def _capture_vector_cluster(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture raft ship placements passed into the vector cluster renderer."""
    captured: list = []

    def _mock_render_vector_cluster(ships, image_size, blur_sigma, scene_scale, **kwargs):
        captured[:] = list(ships)
        return np.zeros((image_size, image_size, 4), dtype=np.uint8)

    monkeypatch.setattr(compose_mod, "_render_vector_raft_cluster", _mock_render_vector_cluster)
    return captured


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

    def test_no_significant_overlap(self, scene) -> None:
        """クラスター内の船同士が大きく重ならないことを検証する。

        raft_tight レイアウトでは意図的に OBB を僅かに重複させて実際の船腹接触を
        実現するため、最大 IoU は 0 にならない。実マスクでの接触を必須にした結果、
        幅 2 px 級の極小船では 1 列の重なりだけで IoU が約 1/3 まで上がり得る。
        そのため上限を 0.35 とし、深いめり込みだけを防ぐ。
        """
        max_iou = 0.0
        for seed in range(30):
            wm = scene["water_mask"].copy()
            oc = np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool)
            rng = random.Random(seed)
            labels = _place_cluster(
                wm, oc, None,
                resolution_m=5.0, rng=rng,
                cluster_size_range=(4, 4),
                blur_sigma=0.0,
                alpha_range=(0.8, 0.9),
                class_id=0,
                image_size=self._IMAGE_SIZE,
                background=scene["background"].copy(),
                length_range=(30.0, 80.0),
                mixed_prob=0.5,
            )
            if len(labels) < 2:
                continue
            polys = [_parse_obb_polygon(l, self._IMAGE_SIZE) for l in labels]
            for a in range(len(polys)):
                for b in range(a + 1, len(polys)):
                    iou = _polygon_iou(polys[a], polys[b])
                    if iou > max_iou:
                        max_iou = iou
        assert max_iou < 0.35, (
            f"Ships overlap too much (max IoU={max_iou:.3f})"
        )

    def test_gap_variety(self, scene) -> None:
        """クラスター間に隙間・接触・ぴったりの3状態が出現することを検証する。"""
        min_gaps: list[float] = []
        for seed in range(50):
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
                length_range=(30.0, 80.0),
                mixed_prob=0.5,
            )
            if len(labels) < 2:
                continue
            polys = [_parse_obb_polygon(l, self._IMAGE_SIZE) for l in labels]
            for a in range(len(polys) - 1):
                gap = _polygon_min_distance(polys[a], polys[a + 1])
                min_gaps.append(gap)

        assert len(min_gaps) >= 10, "テスト成立に必要なサンプル数を得られなかった"
        # Three regimes: tight/touching (gap <= 1), small gap (1 < gap <= 4),
        # visible gap (gap > 4)
        tight = sum(1 for g in min_gaps if g <= 1.5)
        gapped = sum(1 for g in min_gaps if g > 3.0)
        assert tight > 0, "隙間なし（tight/touching）の配置が一度も出現しなかった"
        assert gapped > 0, "隙間あり（gapped）の配置が一度も出現しなかった"

    @pytest.mark.parametrize(
        ("sizes", "description"),
        [
            ([(4, 18), (4, 18), (5, 20)], "small-small"),
            ([(14, 72), (14, 72), (16, 80)], "large-large"),
            ([(4, 18), (4, 18), (14, 72)], "small-large"),
        ],
    )
    def test_tight_cluster_hulls_touch_without_deep_overlap(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
        sizes: list[tuple[int, int]],
        description: str,
    ) -> None:
        """tight クラスターで船腹が接触しつつ過度にめり込まない。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <defs>'
            '    <clipPath id="h">'
            '      <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1"/>'
            '    </clipPath>'
            '  </defs>'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)" stroke="rgb(20,20,20)"/>'
            '</svg>'
        )
        captured = _capture_vector_cluster(monkeypatch)
        monkeypatch.setattr(compose_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(compose_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            compose_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory(sizes),
        )

        rng = _ForcedLayoutRandom(7, "flush", base_angle=0.5)
        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=rng,
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2, f"tight {description} で2隻配置されなかった"
        assert len(captured) == 2, f"tight {description} の vector capture に失敗した"

        gap = captured[0].hull_geom.distance(captured[1].hull_geom)
        overlap_area = captured[0].hull_geom.intersection(captured[1].hull_geom).area
        min_a, max_a = _geometry_projection_extents(captured[0].hull_geom, 1.0, 0.0)
        min_b, _max_b = _geometry_projection_extents(captured[1].hull_geom, 1.0, 0.0)
        penetration_px = max_a - min_b

        assert gap <= 1e-6 or overlap_area > 0.0, f"tight {description} がベクトル幾何で接触していない"
        assert penetration_px <= 1.0, (
            f"tight {description} が row 方向にめり込みすぎている "
            f"(gap={gap:.3f}, overlap_area={overlap_area:.3f}, penetration={penetration_px:.2f})"
        )

    def test_tight_cluster_labels_keep_subpixel_offsets(self, scene, monkeypatch: pytest.MonkeyPatch) -> None:
        """tight クラスターのラベル座標がサブピクセル位置を保持する。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <defs>'
            '    <clipPath id="h">'
            '      <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1"/>'
            '    </clipPath>'
            '  </defs>'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)" stroke="rgb(20,20,20)"/>'
            '</svg>'
        )
        monkeypatch.setattr(compose_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(compose_mod, "find_water_position", lambda *args, **kwargs: (60, 100))

        def _mock_rasterize(svg_text, bw, lh, angle_deg=0.0, supersample=4, exclude_hull=False):
            if exclude_hull:
                return np.zeros((lh, bw, 4), dtype=np.uint8)
            return _make_tapered_hull_rgba(bw, lh)

        monkeypatch.setattr(compose_mod, "rasterize_ship_svg", _mock_rasterize)
        monkeypatch.setattr(
            compose_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(4, 18), (4, 18)]),
        )

        rng = _ForcedLayoutRandom(7, "flush", base_angle=0.5)
        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=rng,
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=scene["background"].copy(),
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2, "tight クラスターで2隻配置されなかった"
        second_poly = _parse_obb_polygon(labels[1], self._IMAGE_SIZE)
        fractions = [
            abs(coord - round(coord))
            for x, y in second_poly
            for coord in (x, y)
        ]
        assert max(fractions) >= 0.05, (
            "tight クラスターの座標がまだ整数ピクセルに量子化されている"
        )

    def test_tight_cluster_final_render_has_no_background_slit(
        self,
        scene,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tight クラスターの最終描画が背景スリットで分断されない。"""
        mock_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <defs>'
            '    <clipPath id="h">'
            '      <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1"/>'
            '    </clipPath>'
            '  </defs>'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(190,190,190)" stroke="rgb(20,20,20)"/>'
            '</svg>'
        )
        monkeypatch.setattr(compose_mod, "_pick_svg", lambda *args, **kwargs: mock_svg)
        monkeypatch.setattr(compose_mod, "find_water_position", lambda *args, **kwargs: (60, 100))
        monkeypatch.setattr(
            compose_mod,
            "_resolve_ship_dimensions",
            _resolve_ship_dimensions_sequence_factory([(6, 24), (6, 24)]),
        )

        background = scene["background"].copy()
        rng = _ForcedLayoutRandom(7, "flush", base_angle=0.0)
        labels = _place_cluster(
            scene["water_mask"],
            np.zeros((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=bool),
            None,
            resolution_m=5.0,
            rng=rng,
            cluster_size_range=(2, 2),
            blur_sigma=0.0,
            alpha_range=(0.8, 0.9),
            class_id=0,
            image_size=self._IMAGE_SIZE,
            background=background,
            length_range=(20.0, 80.0),
            mixed_prob=1.0,
        )

        assert len(labels) == 2, "tight クラスターで2隻配置されなかった"
        ship_mask = np.any(background != scene["background"], axis=2)
        assert _count_connected_components(ship_mask, min_size=20) == 1, (
            "tight クラスターの最終描画が背景スリットで 2 つに分断されている"
        )
    """_worker_init によるワーカープロセス初期化のテスト。"""

    def test_none_svg_dir_sets_none(self) -> None:
        """svg_dir=None のとき _worker_svg_metas が None になる。"""
        import medetect.datagen.compose as compose_mod

        compose_mod._worker_init(None)
        assert compose_mod._worker_svg_metas is None

    def test_svg_dir_populates_metas(self, tmp_path: pathlib.Path) -> None:
        """有効な SVG ディレクトリを渡すと _worker_svg_metas が設定される。"""
        import medetect.datagen.compose as compose_mod
        from medetect.shipgen.gen import generate_ship_svg

        svg_dir = tmp_path / "svgs"
        svg_dir.mkdir()
        rng = random.Random(0)
        for i in range(3):
            (svg_dir / f"ship_{i}.svg").write_text(
                generate_ship_svg("patrol", rng=rng), encoding="utf-8"
            )

        compose_mod._worker_init(svg_dir)
        assert compose_mod._worker_svg_metas is not None
        assert len(compose_mod._worker_svg_metas) == 3
        for meta in compose_mod._worker_svg_metas:
            assert meta.lb_ratio > 0


class TestFalseSourceGrid:
    """_false_source_grid のグリッド計算テスト。"""

    def test_png_exact_tiles(self, tmp_path: pathlib.Path) -> None:
        """PNG 画像のグリッドサイズが正しく計算される。"""
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (1280, 640))
        path = tmp_path / "src.png"
        img.save(path)
        result = _false_source_grid(path, image_size=640, resolution=None, geo_scale=None)
        assert result == (640, 2, 1)

    def test_png_too_small_returns_none(self, tmp_path: pathlib.Path) -> None:
        """小さすぎる PNG は None を返す。"""
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (320, 320))
        path = tmp_path / "small.png"
        img.save(path)
        result = _false_source_grid(path, image_size=640, resolution=None, geo_scale=None)
        assert result is None

    def test_partial_tile_truncated(self, tmp_path: pathlib.Path) -> None:
        """端数はタイルに含まれない（切り捨て）。"""
        from PIL import Image as PILImage
        # 1500px → 1500//640 = 2 cols, 900px → 900//640 = 1 row
        img = PILImage.new("RGB", (1500, 900))
        path = tmp_path / "partial.png"
        img.save(path)
        result = _false_source_grid(path, image_size=640, resolution=None, geo_scale=None)
        assert result == (640, 2, 1)

    def test_geo_scale_applied_to_tif(self, tmp_path: pathlib.Path) -> None:
        """TIFF に geo_scale が適用される。"""
        import rasterio
        from rasterio.transform import from_bounds
        tif_path = tmp_path / "bg.tif"
        size = 3200
        data = np.full((3, size, size), 128, dtype=np.uint8)
        with rasterio.open(
            tif_path, "w", driver="GTiff",
            height=size, width=size, count=3, dtype="uint8",
            transform=from_bounds(0, 0, size, size, size, size),
        ) as dst:
            dst.write(data)
        # geo_scale=2.0 → src_tile = 640*2 = 1280
        result = _false_source_grid(tif_path, image_size=640, resolution=None, geo_scale=2.0)
        assert result is not None
        src_tile, cols, rows = result
        assert src_tile == 1280
        assert cols == size // 1280
        assert rows == size // 1280


class TestGenerateFalseNegatives:
    """generate_false_negatives の機能テスト。"""

    @staticmethod
    def _make_source(path: pathlib.Path, width: int, height: int, color: tuple) -> None:
        from PIL import Image as PILImage
        PILImage.new("RGB", (width, height), color=color).save(path)

    def test_writes_images_and_empty_labels(self, tmp_path: pathlib.Path) -> None:
        """False negative タイルと空ラベルが正しく書き出される。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        # 2 sources, each 1280×640 → 2 tiles each (capacity=4)
        self._make_source(false_dir / "a.png", 1280, 640, (80, 100, 120))
        self._make_source(false_dir / "b.png", 1280, 640, (60, 80, 100))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        n = generate_false_negatives(
            false_dir, out_dir, count=3, image_size=640, rng=rng, start_index=0
        )
        assert n == 3
        for i in range(3):
            assert (out_dir / "images" / "train" / f"{i:06d}.png").exists()
            lbl = out_dir / "labels" / "train" / f"{i:06d}.txt"
            assert lbl.exists()
            assert lbl.read_text(encoding="utf-8") == ""

    def test_start_index_offsets_names(self, tmp_path: pathlib.Path) -> None:
        """start_index によりファイル番号がオフセットされる。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        self._make_source(false_dir / "a.png", 1280, 640, (80, 100, 120))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        n = generate_false_negatives(
            false_dir, out_dir, count=2, image_size=640, rng=rng, start_index=100
        )
        assert n == 2
        assert (out_dir / "images" / "train" / "000100.png").exists()
        assert (out_dir / "images" / "train" / "000101.png").exists()
        assert not (out_dir / "images" / "train" / "000000.png").exists()

    def test_no_overlap_exhausts_grid(self, tmp_path: pathlib.Path) -> None:
        """1画像から全タイルを要求しても重複なく書き出せる。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        # 4×4 grid = 16 non-overlapping tiles
        self._make_source(false_dir / "a.png", 2560, 2560, (80, 100, 120))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        n = generate_false_negatives(
            false_dir, out_dir, count=16, image_size=640, rng=rng
        )
        assert n == 16

    def test_even_distribution_multi_source(self, tmp_path: pathlib.Path) -> None:
        """複数ソース間で均等配分される（1枚への集中を防ぐ）。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        # 4 sources, each with exactly 1 tile (640×640). Request 4.
        # With max_per_source = ceil(4/4) = 1, must use all 4 sources.
        for i in range(4):
            self._make_source(false_dir / f"src{i}.png", 640, 640, (50 + i * 20, 60, 70))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        n = generate_false_negatives(
            false_dir, out_dir, count=4, image_size=640, rng=rng
        )
        # Each source has exactly 1 tile, so we need all 4 to reach count=4.
        assert n == 4

    def test_empty_dir_raises(self, tmp_path: pathlib.Path) -> None:
        """ソース画像がないディレクトリは FileNotFoundError を送出する。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        with pytest.raises(FileNotFoundError):
            generate_false_negatives(
                false_dir, out_dir, count=2, image_size=640, rng=rng
            )

    def test_tile_size_matches_image_size(self, tmp_path: pathlib.Path) -> None:
        """書き出された PNG のサイズが image_size と一致する。"""
        from PIL import Image as PILImage
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        self._make_source(false_dir / "a.png", 1280, 640, (80, 100, 120))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(0)
        generate_false_negatives(
            false_dir, out_dir, count=1, image_size=320, rng=rng
        )
        with PILImage.open(out_dir / "images" / "train" / "000000.png") as img:
            assert img.size == (320, 320)

    def test_count_met_when_capacity_insufficient(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """容量不足でも要求枚数分が出力され、警告が発される。"""
        import logging
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        # 1 source → 2 non-overlapping tiles (1280 // 640 * 640 // 640 = 2×1 = 2)
        self._make_source(false_dir / "a.png", 1280, 640, (80, 100, 120))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(7)
        with caplog.at_level(logging.WARNING, logger="medetect.datagen.compose"):
            n = generate_false_negatives(
                false_dir, out_dir, count=5, image_size=640, rng=rng
            )
        assert n == 5
        assert (out_dir / "images" / "train" / "000004.png").exists()
        assert any("repeated" in r.message.lower() for r in caplog.records)

    def test_count_met_multi_source_capacity_insufficient(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """複数ソースでも容量不足時に全枚数が出力される。"""
        import logging
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        # 2 sources × 1 tile = 2 total capacity; request 7
        self._make_source(false_dir / "a.png", 640, 640, (80, 100, 120))
        self._make_source(false_dir / "b.png", 640, 640, (60, 80, 100))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(3)
        with caplog.at_level(logging.WARNING, logger="medetect.datagen.compose"):
            n = generate_false_negatives(
                false_dir, out_dir, count=7, image_size=640, rng=rng
            )
        assert n == 7
        assert (out_dir / "images" / "train" / "000006.png").exists()
        assert any("repeated" in r.message.lower() for r in caplog.records)


class TestFalseRatioSplit:
    """false_ratio による合成/偽陰性の枚数分割のテスト。"""

    def test_ratio_zero_no_false(self) -> None:
        """false_ratio=0.0 のとき偽陰性は 0 枚。"""
        # count=10, false_ratio=0.0 → synth=10, false=0
        total = 10
        false_ratio = 0.0
        false_count = round(total * false_ratio)
        synth_count = total - false_count
        assert synth_count == 10
        assert false_count == 0

    def test_ratio_09_gives_correct_split(self) -> None:
        """count=100, false_ratio=0.9 → 合成10+偽陰性90=計100。"""
        total = 100
        false_ratio = 0.9
        false_count = round(total * false_ratio)
        synth_count = total - false_count
        assert synth_count == 10
        assert false_count == 90
        assert synth_count + false_count == total

    def test_ratio_02_gives_correct_split(self) -> None:
        """count=100, false_ratio=0.2 → 合成80+偽陰性20=計100。"""
        total = 100
        false_ratio = 0.2
        false_count = round(total * false_ratio)
        synth_count = total - false_count
        assert synth_count == 80
        assert false_count == 20
        assert synth_count + false_count == total

    def test_ratio_05_splits_evenly(self) -> None:
        """count=100, false_ratio=0.5 → 合成50+偽陰性50=計100。"""
        total = 100
        false_ratio = 0.5
        false_count = round(total * false_ratio)
        synth_count = total - false_count
        assert synth_count == 50
        assert false_count == 50
        assert synth_count + false_count == total

    def test_total_preserved_for_various_counts(self) -> None:
        """様々な count/ratio で合計が常に count に等しい。"""
        cases = [
            (10, 0.3),
            (7, 0.5),
            (1000, 0.1),
            (3, 0.9),
        ]
        for total, ratio in cases:
            false_count = round(total * ratio)
            synth_count = total - false_count
            assert synth_count + false_count == total, (
                f"count={total}, ratio={ratio}: {synth_count}+{false_count}≠{total}"
            )


class TestDatagenCli:
    """datagen CLI の公開オプション整合を検証する。"""

    def test_help_omits_removed_debug_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--help から削除対象オプションが消え、説明文が現行仕様に一致する。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        monkeypatch.setattr(sys, "argv", ["medetect.datagen", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            datagen_main.main()

        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out

        assert "--force_tight_clusters" not in help_text
        assert "--debug_bg_color" not in help_text
        assert "--disable-water-tint" not in help_text
        assert "placement events per image" in help_text
        assert "single ships only" in help_text
