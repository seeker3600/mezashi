"""medetect.xview.slice の純粋計算ヘルパーに対する単体テスト。"""

from __future__ import annotations

import math

import pytest

import medetect.xview.slice as _m
from medetect.xview.slice import (
    _choose_resolution,
    _compute_geo_resolution,
    _compute_resample_params,
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
# _compute_resample_params
# ---------------------------------------------------------------------------

class TestComputeResampleParams:
    """_compute_resample_params のテスト。"""

    def test_same_scale(self) -> None:
        """target_res == native_res のとき scale=1 → サイズ変化なし。"""
        result = _compute_resample_params(
            native_res=1.0, src_width=1280, src_height=1280,
            image_size=640, resolution=1.0,
        )
        assert result is not None
        target_res, new_w, new_h = result
        assert target_res == pytest.approx(1.0)
        assert new_w == 1280
        assert new_h == 1280

    def test_finer_resolution_doubles_size(self) -> None:
        """target_res が native_res の半分 → 幅・高さが 2 倍になる。"""
        result = _compute_resample_params(
            native_res=1.0, src_width=800, src_height=800,
            image_size=640, resolution=0.5,
        )
        assert result is not None
        _, new_w, new_h = result
        assert new_w == 1600
        assert new_h == 1600

    def test_coarser_resolution_capped_by_max(self) -> None:
        """指定 resolution が粗すぎる場合は max_target_res に切り詰められる。
        640x640 画像, image_size=640 → max_target=native_res → scale=1。
        """
        result = _compute_resample_params(
            native_res=1.0, src_width=640, src_height=640,
            image_size=640, resolution=99.0,
        )
        assert result is not None
        target_res, new_w, new_h = result
        assert target_res == pytest.approx(1.0)
        assert new_w == 640
        assert new_h == 640

    def test_non_square_src(self) -> None:
        """非正方形画像でも min_dim を基準に計算し両辺とも image_size 以上になる。"""
        # native=0.3, src=2000x1280, image_size=640
        # min_dim=1280, max_target=0.3*1280/640=0.6
        # resolution=0.3 < 0.6 → target=0.3, scale=1.0
        result = _compute_resample_params(
            native_res=0.3, src_width=2000, src_height=1280,
            image_size=640, resolution=0.3,
        )
        assert result is not None
        _, new_w, new_h = result
        assert new_w >= 640
        assert new_h >= 640

    def test_returns_none_when_scale_too_small(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_choose_resolution が強制的に大きな値を返す場合、None になる。
        scale<1 → リサンプリング後サイズが image_size 未満 → None。
        """
        monkeypatch.setattr(_m, "_choose_resolution", lambda *a, **kw: 100.0)
        result = _compute_resample_params(
            native_res=0.1, src_width=100, src_height=100,
            image_size=640, resolution=100.0,
        )
        assert result is None

    def test_tuple_resolution_stays_in_expected_range(self) -> None:
        """範囲指定 resolution を渡しても結果が合理的な範囲に収まる。"""
        # native=1.0, src=2000x2000, resolution=(0.5, 1.0)
        # scale ∈ [1.0, 2.0] → new_dim ∈ [2000, 4000]
        for _ in range(30):
            result = _compute_resample_params(
                native_res=1.0, src_width=2000, src_height=2000,
                image_size=640, resolution=(0.5, 1.0),
            )
            assert result is not None
            _, new_w, new_h = result
            assert 2000 <= new_w <= 4001
            assert 2000 <= new_h <= 4001

    def test_target_res_in_result(self) -> None:
        """戻り値の target_res が指定 resolution と一致する（固定値の場合）。"""
        result = _compute_resample_params(
            native_res=0.5, src_width=1280, src_height=1280,
            image_size=640, resolution=0.25,
        )
        assert result is not None
        target_res, _, _ = result
        assert target_res == pytest.approx(0.25)
