from __future__ import annotations

import math

import pytest

from medetect.datagen.obb import compute_obb_corners, format_obb_label


class TestComputeObbCorners:
    def test_axis_aligned(self) -> None:
        """回転なしのOBBは軸揃いになる。"""
        corners = compute_obb_corners(50.0, 50.0, 10.0, 20.0, 0.0)
        assert len(corners) == 4
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        assert min(xs) == pytest.approx(45.0)
        assert max(xs) == pytest.approx(55.0)
        assert min(ys) == pytest.approx(40.0)
        assert max(ys) == pytest.approx(60.0)

    def test_rotated_90(self) -> None:
        """90度回転でOBBの幅と高さが入れ替わる。"""
        corners = compute_obb_corners(50.0, 50.0, 10.0, 20.0, math.pi / 2)
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        assert min(xs) == pytest.approx(40.0, abs=0.1)
        assert max(xs) == pytest.approx(60.0, abs=0.1)
        assert min(ys) == pytest.approx(45.0, abs=0.1)
        assert max(ys) == pytest.approx(55.0, abs=0.1)

    def test_corners_form_rectangle(self) -> None:
        """4点が長方形を構成する（対角線の長さが等しい）。"""
        corners = compute_obb_corners(100.0, 100.0, 20.0, 40.0, 0.7)
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
        assert len(parts) == 9
        assert float(parts[1]) == pytest.approx(0.1)
        assert float(parts[2]) == pytest.approx(0.2)

    def test_coordinates_normalized(self) -> None:
        """座標が画像サイズで正規化される。"""
        corners = [(100.0, 200.0), (300.0, 200.0), (300.0, 400.0), (100.0, 400.0)]
        label = format_obb_label(1, corners, img_w=640, img_h=640)
        parts = label.split()
        for idx in range(1, 9):
            val = float(parts[idx])
            assert 0.0 <= val <= 1.0