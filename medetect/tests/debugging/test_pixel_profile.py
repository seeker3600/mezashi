"""Tests for medetect.debugging.pixel_profile."""

from __future__ import annotations

import numpy as np
import pytest

from medetect.debugging.pixel_profile import (
    extract_line_profile,
    normalize_values,
    resolve_coords,
)


class TestExtractLineProfile:
    def test_horizontal_line_grayscale(self) -> None:
        """水平方向の直線でピクセル値が左→右の順に返る。"""
        arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
        pos, vals = extract_line_profile(arr, 0, 5, 9, 5)
        assert vals[0] == 50
        assert vals[-1] == 59
        assert len(pos) == len(vals)
        assert pos[0] == pytest.approx(0.0)
        assert pos[-1] == pytest.approx(9.0)

    def test_vertical_line_grayscale(self) -> None:
        """垂直方向の直線でピクセル値が上→下の順に返る。"""
        arr = np.zeros((10, 10), dtype=np.uint8)
        arr[:, 3] = np.arange(0, 100, 10, dtype=np.uint8)
        pos, vals = extract_line_profile(arr, 3, 0, 3, 9)
        assert vals[0] == 0
        assert vals[-1] == 90

    def test_single_point_line(self) -> None:
        """始点と終点が一致する場合も落ちない。"""
        arr = np.full((5, 5), 42, dtype=np.uint8)
        _pos, vals = extract_line_profile(arr, 2, 2, 2, 2)
        assert vals[0] == 42

    def test_rgb_image_shape(self) -> None:
        """カラー画像は (N, 3) の形で返る。"""
        arr = np.zeros((20, 20, 3), dtype=np.uint8)
        arr[:, :, 0] = 100
        arr[:, :, 1] = 150
        arr[:, :, 2] = 200
        _pos, vals = extract_line_profile(arr, 0, 0, 19, 0)
        assert vals.shape == (20, 3)
        assert (vals[:, 0] == 100).all()
        assert (vals[:, 1] == 150).all()
        assert (vals[:, 2] == 200).all()

    def test_out_of_bounds_clamped(self) -> None:
        """座標が画像境界を超えてもクリップされてエラーにならない。"""
        arr = np.full((10, 10), 77, dtype=np.uint8)
        _pos, vals = extract_line_profile(arr, -5, 5, 20, 5)
        assert (vals == 77).all()


class TestNormalizeValues:
    def test_grayscale_max_becomes_one(self) -> None:
        """255 が 1.0 に正規化される。"""
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = normalize_values(arr)
        assert result.dtype == np.float32
        assert result[2] == pytest.approx(1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(128 / 255)

    def test_rgb_shape_preserved(self) -> None:
        """カラー配列のシェイプが維持される。"""
        arr = np.full((10, 3), 255, dtype=np.uint8)
        result = normalize_values(arr)
        assert result.shape == (10, 3)
        assert result == pytest.approx(np.ones((10, 3), dtype=np.float32))

    def test_zero_stays_zero(self) -> None:
        """0 は 0.0 のまま。"""
        arr = np.zeros((5,), dtype=np.uint8)
        result = normalize_values(arr)
        assert result == pytest.approx(np.zeros(5, dtype=np.float32))


class TestResolveCoords:
    def test_normalized_all_in_01(self) -> None:
        """全座標が [0,1] → 正規化として解釈される。"""
        px0, py0, px1, py1 = resolve_coords((0.0, 0.0, 1.0, 1.0), 640, 640)
        assert px0 == 0
        assert py0 == 0
        assert px1 == 639
        assert py1 == 639

    def test_absolute_when_over_1(self) -> None:
        """いずれかの座標が 1 を超えるとき絶対ピクセルとして解釈される。"""
        px0, py0, px1, py1 = resolve_coords((10.0, 20.0, 200.0, 300.0), 640, 640)
        assert px0 == 10
        assert py0 == 20
        assert px1 == 200
        assert py1 == 300

    def test_normalized_center(self) -> None:
        """中心点 (0.5, 0.5) が画像中央ピクセルにマップされる。"""
        px0, py0, px1, py1 = resolve_coords((0.5, 0.5, 0.5, 0.5), 100, 200)
        assert px0 == 50
        assert py0 == 100