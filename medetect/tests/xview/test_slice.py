"""medetect.xview.slice の純粋計算ヘルパーに対する単体テスト。"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import medetect.xview.slice as _m
from medetect.xview.slice import (
    _choose_resolution,
    _clip_labels_to_window,
    _compute_geo_resolution,
    _compute_native_tile_size,
    _compute_tile_transform,
    _iter_tile_windows,
    _parse_yolo_labels,
)


# ---------------------------------------------------------------------------
# _compute_geo_resolution
# ---------------------------------------------------------------------------

class TestComputeGeoResolution:
    """_compute_geo_resolution のテスト。"""

    def test_projected_returns_average(self) -> None:
        """投影座標系（メートル単位）ではそのまま平均を返す。"""
        result = _compute_geo_resolution(2.0, 4.0, is_geographic=False)
        assert result == pytest.approx(3.0)

    def test_projected_equal_pixels(self) -> None:
        """正方ピクセルでは res_x そのものを返す。"""
        result = _compute_geo_resolution(0.5, 0.5, is_geographic=False)
        assert result == pytest.approx(0.5)

    def test_geographic_equator(self) -> None:
        """地理座標系・赤道 (center_lat=0) では度あたりメートルをそのまま適用する。"""
        deg = 1e-4  # 0.0001 度
        expected = deg * 111320.0  # 経緯ともに同じスケール
        result = _compute_geo_resolution(deg, deg, is_geographic=True, center_lat=0.0)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_geographic_45deg(self) -> None:
        """地理座標系・北緯45度では経度方向に cos(45°) の縮小がかかる。"""
        deg = 1e-4
        meters_lat = deg * 111320.0
        meters_lon = deg * 111320.0 * math.cos(math.radians(45.0))
        expected = (meters_lon + meters_lat) / 2.0
        result = _compute_geo_resolution(deg, deg, is_geographic=True, center_lat=45.0)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_geographic_non_square_pixel(self) -> None:
        """res_x と res_y が異なる場合でも正しく計算する。"""
        res_x, res_y = 1e-4, 2e-4
        meters_lon = res_x * 111320.0 * math.cos(math.radians(30.0))
        meters_lat = res_y * 111320.0
        expected = (meters_lon + meters_lat) / 2.0
        result = _compute_geo_resolution(res_x, res_y, is_geographic=True, center_lat=30.0)
        assert result == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# _choose_resolution
# ---------------------------------------------------------------------------

class TestChooseResolution:
    """_choose_resolution のテスト。"""

    def test_fixed_no_cap(self) -> None:
        """固定値・上限なし → そのまま返す。"""
        assert _choose_resolution(1.5) == pytest.approx(1.5)

    def test_fixed_below_cap(self) -> None:
        """固定値が上限以下 → そのまま返す。"""
        assert _choose_resolution(1.0, max_resolution=2.0) == pytest.approx(1.0)

    def test_fixed_above_cap(self) -> None:
        """固定値が上限を超えている → 上限に切り詰める。"""
        assert _choose_resolution(3.0, max_resolution=2.0) == pytest.approx(2.0)

    def test_fixed_equal_cap(self) -> None:
        """固定値 == 上限 → 上限を返す。"""
        assert _choose_resolution(2.0, max_resolution=2.0) == pytest.approx(2.0)

    def test_tuple_within_range(self) -> None:
        """範囲指定では結果が [low, high] に収まる。"""
        results = [_choose_resolution((1.0, 3.0)) for _ in range(200)]
        assert all(1.0 <= r <= 3.0 for r in results)

    def test_tuple_cap_clips_high(self) -> None:
        """範囲指定・上限あり → high が上限に切り詰められた範囲に収まる。"""
        results = [_choose_resolution((1.0, 5.0), max_resolution=3.0) for _ in range(200)]
        assert all(1.0 <= r <= 3.0 for r in results)

    def test_tuple_cap_below_low_returns_cap(self) -> None:
        """上限が low を下回る → high == low == max_resolution 固定値を返す。"""
        results = [_choose_resolution((2.0, 5.0), max_resolution=1.0) for _ in range(50)]
        assert all(r == pytest.approx(1.0) for r in results)

    def test_tuple_no_cap_samples_whole_range(self) -> None:
        """上限なしの場合、低値・高値付近のサンプルが得られる（確率的な健全性確認）。"""
        results = [_choose_resolution((0.0, 10.0)) for _ in range(500)]
        assert min(results) < 1.0
        assert max(results) > 9.0


# ---------------------------------------------------------------------------
# _compute_native_tile_size
# ---------------------------------------------------------------------------

class TestComputeNativeTileSize:
    """_compute_native_tile_size のテスト。"""

    def test_same_scale(self) -> None:
        """target_res == native_res → native_tile_size == image_size。"""
        result = _compute_native_tile_size(
            native_res=1.0, src_width=1280, src_height=1280,
            image_size=640, resolution=1.0,
        )
        assert result is not None
        target_res, tile_size = result
        assert target_res == pytest.approx(1.0)
        assert tile_size == pytest.approx(640.0)

    def test_finer_resolution_smaller_window(self) -> None:
        """target_res が native_res の半分 → ウィンドウは image_size の半分。"""
        result = _compute_native_tile_size(
            native_res=1.0, src_width=800, src_height=800,
            image_size=640, resolution=0.5,
        )
        assert result is not None
        _, tile_size = result
        assert tile_size == pytest.approx(320.0)

    def test_coarser_resolution_capped_by_max(self) -> None:
        """resolution が粗すぎる場合は max_target_res に切り詰められる。"""
        result = _compute_native_tile_size(
            native_res=1.0, src_width=640, src_height=640,
            image_size=640, resolution=99.0,
        )
        assert result is not None
        target_res, tile_size = result
        assert target_res == pytest.approx(1.0)
        assert tile_size == pytest.approx(640.0)

    def test_non_square_src(self) -> None:
        """非正方形画像でも min_dim を基準にタイルサイズが決まる。"""
        result = _compute_native_tile_size(
            native_res=0.3, src_width=2000, src_height=1280,
            image_size=640, resolution=0.3,
        )
        assert result is not None
        _, tile_size = result
        assert tile_size == pytest.approx(640.0)

    def test_returns_none_when_scale_too_small(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_choose_resolution が大きな値を返す場合、None になる。"""
        monkeypatch.setattr(_m, "_choose_resolution", lambda *a, **kw: 100.0)
        result = _compute_native_tile_size(
            native_res=0.1, src_width=100, src_height=100,
            image_size=640, resolution=100.0,
        )
        assert result is None

    def test_tuple_resolution_stays_in_expected_range(self) -> None:
        """範囲指定 resolution を渡しても結果が合理的な範囲に収まる。"""
        for _ in range(30):
            result = _compute_native_tile_size(
                native_res=1.0, src_width=2000, src_height=2000,
                image_size=640, resolution=(0.5, 1.0),
            )
            assert result is not None
            _, tile_size = result
            # tile = image_size * target_res / native_res
            # target_res ∈ [0.5, 1.0] → tile ∈ [320, 640]
            assert 320.0 <= tile_size <= 640.0 + 1e-9

    def test_target_res_in_result(self) -> None:
        """戻り値の target_res が指定 resolution と一致する（固定値の場合）。"""
        result = _compute_native_tile_size(
            native_res=0.5, src_width=1280, src_height=1280,
            image_size=640, resolution=0.25,
        )
        assert result is not None
        target_res, _ = result
        assert target_res == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _iter_tile_windows
# ---------------------------------------------------------------------------

class TestIterTileWindows:
    """_iter_tile_windows のテスト。"""

    def test_single_tile_exact_fit(self) -> None:
        """画像がタイルと同じサイズ → 1 タイル。"""
        tiles = _iter_tile_windows(640, 640, 640.0, 0.0)
        assert len(tiles) == 1
        row, col, col_off, row_off = tiles[0]
        assert (row, col) == (0, 0)
        assert col_off == pytest.approx(0.0)
        assert row_off == pytest.approx(0.0)

    def test_two_by_two_no_overlap(self) -> None:
        """画像がタイル 2×2 にぴったり → 4 タイル。"""
        tiles = _iter_tile_windows(1280, 1280, 640.0, 0.0)
        assert len(tiles) == 4
        offsets = [(t[2], t[3]) for t in tiles]
        assert any(c == pytest.approx(0.0) and r == pytest.approx(0.0) for c, r in offsets)
        assert any(c == pytest.approx(640.0) for c, _ in offsets)
        assert any(r == pytest.approx(640.0) for _, r in offsets)

    def test_overlap_generates_more_tiles(self) -> None:
        """overlap > 0 ではタイル数が増える。"""
        tiles_no = _iter_tile_windows(1280, 1280, 640.0, 0.0)
        tiles_ov = _iter_tile_windows(1280, 1280, 640.0, 0.5)
        assert len(tiles_ov) > len(tiles_no)

    def test_edge_clamping(self) -> None:
        """端のタイルは画像境界からはみ出さないようにクランプされる。"""
        tiles = _iter_tile_windows(1000, 1000, 640.0, 0.0)
        for _row, _col, col_off, row_off in tiles:
            assert col_off + 640.0 <= 1000.0 + 1e-9
            assert row_off + 640.0 <= 1000.0 + 1e-9

    def test_tile_too_large_returns_empty(self) -> None:
        """タイルサイズが画像より大きい → 空リスト。"""
        tiles = _iter_tile_windows(500, 500, 640.0, 0.0)
        assert tiles == []

    def test_non_square_image(self) -> None:
        """横長画像でも正しくタイルが生成される。"""
        tiles = _iter_tile_windows(2000, 800, 640.0, 0.0)
        assert len(tiles) >= 2  # at least 2 columns
        for _row, _col, col_off, row_off in tiles:
            assert col_off >= 0.0
            assert row_off >= 0.0
            assert col_off + 640.0 <= 2000.0 + 1e-9
            assert row_off + 640.0 <= 800.0 + 1e-9


# ---------------------------------------------------------------------------
# _parse_yolo_labels
# ---------------------------------------------------------------------------

class TestParseYoloLabels:
    """_parse_yolo_labels のテスト。"""

    def test_basic(self, tmp_path: Path) -> None:
        """正常なラベルファイルを解析する。"""
        from pathlib import Path  # noqa: F811

        p = tmp_path / "labels.txt"
        p.write_text("0 0.5 0.5 0.1 0.2\n1 0.3 0.7 0.05 0.05\n")
        labels = _parse_yolo_labels(p)
        assert len(labels) == 2
        assert labels[0] == (0, 0.5, 0.5, 0.1, 0.2)
        assert labels[1] == (1, 0.3, 0.7, 0.05, 0.05)

    def test_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        """存在しないファイル → 空リスト。"""
        from pathlib import Path  # noqa: F811

        labels = _parse_yolo_labels(tmp_path / "missing.txt")
        assert labels == []

    def test_empty_file(self, tmp_path: Path) -> None:
        """空ファイル → 空リスト。"""
        from pathlib import Path  # noqa: F811

        p = tmp_path / "empty.txt"
        p.write_text("")
        labels = _parse_yolo_labels(p)
        assert labels == []


# ---------------------------------------------------------------------------
# _clip_labels_to_window
# ---------------------------------------------------------------------------

class TestClipLabelsToWindow:
    """_clip_labels_to_window のテスト。"""

    def test_label_fully_inside(self) -> None:
        """タイル内に完全に収まるラベル → タイル座標に変換される。"""
        # Label: center (100, 100), size 50x50 in 1000x1000 image
        labels = [(0, 0.1, 0.1, 0.05, 0.05)]
        result = _clip_labels_to_window(
            labels, 0.0, 0.0, 500.0, 1000, 1000,
        )
        assert len(result) == 1
        cls_id, xc, yc, w, h = result[0]
        assert cls_id == 0
        assert xc == pytest.approx(0.2)
        assert yc == pytest.approx(0.2)
        assert w == pytest.approx(0.1)
        assert h == pytest.approx(0.1)

    def test_label_outside(self) -> None:
        """タイル外のラベル → 除外される。"""
        labels = [(0, 0.9, 0.9, 0.05, 0.05)]
        result = _clip_labels_to_window(
            labels, 0.0, 0.0, 500.0, 1000, 1000,
        )
        assert len(result) == 0

    def test_label_partially_clipped(self) -> None:
        """タイル境界をまたぐラベル → クリッピングされる。"""
        # Label center at (450, 250) with size (200, 100) in 1000x1000
        # Abs box: x=[350, 550], y=[200, 300]
        # Tile: [0, 500] x [0, 500]
        # Clipped: x=[350, 500], y=[200, 300]
        labels = [(0, 0.45, 0.25, 0.2, 0.1)]
        result = _clip_labels_to_window(
            labels, 0.0, 0.0, 500.0, 1000, 1000,
        )
        assert len(result) == 1
        cls_id, xc, yc, w, h = result[0]
        assert cls_id == 0
        assert xc == pytest.approx(0.85)
        assert yc == pytest.approx(0.5)
        assert w == pytest.approx(0.3)
        assert h == pytest.approx(0.2)

    def test_min_area_ratio_filters(self) -> None:
        """クリッピング後の面積比が閾値未満のラベルは除外される。"""
        # Label at edge: only 5% inside the tile
        # center (490, 250), size (200, 100) → box x=[390, 590], y=[200, 300]
        # Tile [0, 500], clip x=[390, 500]=110, y=[200, 300]=100
        # area_ratio = (110*100) / (200*100) = 0.55
        labels = [(0, 0.49, 0.25, 0.2, 0.1)]
        # With min_area_ratio=0.6 → excluded
        result = _clip_labels_to_window(
            labels, 0.0, 0.0, 500.0, 1000, 1000,
            min_area_ratio=0.6,
        )
        assert len(result) == 0
        # With min_area_ratio=0.5 → included
        result = _clip_labels_to_window(
            labels, 0.0, 0.0, 500.0, 1000, 1000,
            min_area_ratio=0.5,
        )
        assert len(result) == 1

    def test_non_origin_window(self) -> None:
        """原点以外のウィンドウでもラベルが正しく変換される。"""
        # Label center at (600, 600), size (100, 100) in 1000x1000
        # Tile window: col_off=500, row_off=500, size=500
        # Abs box: x=[550, 650], y=[550, 650]
        # Clip to [500, 1000]: no clipping needed
        # Tile coords: cx = (600-500)/500 = 0.2, cy = 0.2, w = 0.2, h = 0.2
        labels = [(1, 0.6, 0.6, 0.1, 0.1)]
        result = _clip_labels_to_window(
            labels, 500.0, 500.0, 500.0, 1000, 1000,
        )
        assert len(result) == 1
        cls_id, xc, yc, w, h = result[0]
        assert cls_id == 1
        assert xc == pytest.approx(0.2)
        assert yc == pytest.approx(0.2)
        assert w == pytest.approx(0.2)
        assert h == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# _compute_tile_transform
# ---------------------------------------------------------------------------

class TestComputeTileTransform:
    """_compute_tile_transform のテスト。"""

    def test_identity_at_origin(self) -> None:
        """原点タイル・等倍ではソース変換と一致する。"""
        from rasterio.transform import Affine

        src = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
        result = _compute_tile_transform(src, 0.0, 0.0, 640.0, 640)
        assert result.a == pytest.approx(1.0)
        assert result.e == pytest.approx(-1.0)
        assert result.c == pytest.approx(100.0)
        assert result.f == pytest.approx(200.0)

    def test_offset_tile(self) -> None:
        """オフセットされたタイルの原点が正しく計算される。"""
        from rasterio.transform import Affine

        src = Affine(0.5, 0.0, 100.0, 0.0, -0.5, 200.0)
        result = _compute_tile_transform(src, 100.0, 200.0, 640.0, 640)
        assert result.c == pytest.approx(100.0 + 100 * 0.5)  # 150.0
        assert result.f == pytest.approx(200.0 + 200 * (-0.5))  # 100.0

    def test_resampling_scale(self) -> None:
        """リサンプリングでピクセルサイズが適切にスケールされる。"""
        from rasterio.transform import Affine

        src = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        # native_tile_size=1280 → image_size=640: 2x downsampling
        result = _compute_tile_transform(src, 0.0, 0.0, 1280.0, 640)
        assert result.a == pytest.approx(2.0)
        assert result.e == pytest.approx(-2.0)

    def test_non_zero_rotation(self) -> None:
        """回転係数がある場合でも正しく計算される。"""
        from rasterio.transform import Affine

        src = Affine(0.5, 0.1, 1000.0, 0.1, -0.5, 2000.0)
        result = _compute_tile_transform(src, 100.0, 50.0, 320.0, 640)
        scale = 320.0 / 640
        expected_c = 1000.0 + 100.0 * 0.5 + 50.0 * 0.1
        expected_f = 2000.0 + 100.0 * 0.1 + 50.0 * (-0.5)
        assert result.a == pytest.approx(0.5 * scale)
        assert result.b == pytest.approx(0.1 * scale)
        assert result.c == pytest.approx(expected_c)
        assert result.d == pytest.approx(0.1 * scale)
        assert result.e == pytest.approx(-0.5 * scale)
        assert result.f == pytest.approx(expected_f)
