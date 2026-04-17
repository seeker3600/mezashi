from __future__ import annotations

import numpy as np
import pytest

from medetect.datagen.render import (
    extract_hull_polygon,
    parse_color,
    parse_svg_metadata,
    rasterize_ship_svg,
)


class TestParseColor:
    def test_rgb_format(self) -> None:
        """rgb(r,g,b) 形式をパースする。"""
        assert parse_color("rgb(128,64,32)") == (128, 64, 32, 255)

    def test_rgb_with_high_values(self) -> None:
        """rgb上限値をパースする。"""
        assert parse_color("rgb(255,255,255)") == (255, 255, 255, 255)

    def test_rgba_format(self) -> None:
        """rgba(r,g,b,a) 形式をパースする。"""
        assert parse_color("rgba(0,0,0,0.5)") == (0, 0, 0, 128)

    def test_rgba_opaque(self) -> None:
        """rgba with alpha=1.0 returns fully opaque."""
        assert parse_color("rgba(100,200,50,1.0)") == (100, 200, 50, 255)

    def test_none_returns_fallback(self) -> None:
        """未知の形式ではフォールバック色を返す。"""
        r, g, b, a = parse_color("unknown")
        assert 0 <= r <= 255


class TestParseSvgMetadata:
    def test_extracts_class_and_ratio(self) -> None:
        """data-ship-class と data-lb-ratio を抽出する。"""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 1 8.5" '
            'data-ship-class="destroyer" data-lb-ratio="8.5">'
            "</svg>"
        )
        cls, ratio = parse_svg_metadata(svg)
        assert cls == "destroyer"
        assert ratio == pytest.approx(8.5)

    def test_missing_attributes_returns_defaults(self) -> None:
        """属性がない場合デフォルト値を返す。"""
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 5"></svg>'
        cls, ratio = parse_svg_metadata(svg)
        assert cls == "unknown"
        assert ratio == pytest.approx(5.0)


class TestExtractHullPolygon:
    def test_prefers_clippath_hull_polygon(self) -> None:
        """clipPath 内の hull polygon を優先して抽出する。"""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <defs>'
            '    <clipPath id="h">'
            '      <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1"/>'
            '    </clipPath>'
            '  </defs>'
            '  <polygon points="0,0 1,0 1,4 0,4" fill="rgb(20,20,20)"/>'
            '</svg>'
        )
        points = extract_hull_polygon(svg)
        assert points == pytest.approx(
            [(0.5, 0.0), (1.0, 1.0), (1.0, 3.0), (0.5, 4.0), (0.0, 3.0), (0.0, 1.0)]
        )


class TestRasterizeShipSvg:
    _SIMPLE_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4" '
        'data-ship-class="test" data-lb-ratio="4">'
        '  <polygon points="0.5,0 1,2 0.5,4 0,2" fill="rgb(128,128,128)"/>'
        "</svg>"
    )

    def test_returns_rgba_shape(self) -> None:
        """RGBA 4チャンネルの配列を返す。"""
        result = rasterize_ship_svg(self._SIMPLE_SVG, 10, 40)
        assert result.shape == (40, 10, 4)
        assert result.dtype == np.uint8

    def test_hull_pixels_opaque(self) -> None:
        """船体中央のピクセルが完全に不透明（alpha==255）。"""
        result = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80)
        center_alpha = result[40, 10, 3]
        assert center_alpha == 255

    def test_rgba_overlay_hull_stays_opaque(self) -> None:
        """rgba()オーバーレイを重ねた後も船体ベースのalphaが255のまま。

        Porter-Duff 'over' 合成では不透明なベースの上に半透明レイヤーを重ねても
        アルファ値は 1.0 を保つ。以前の実装は直接上書きを行ったため割れていた。
        """
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">'
            # 不透明ベース
            '  <rect x="0" y="0" width="1" height="1" fill="rgb(100,80,60)"/>'
            # 半透明オーバーレイ（エッジダーケン相当）
            '  <rect x="0" y="0" width="1" height="1" fill="rgba(0,0,0,0.40)"/>'
            "</svg>"
        )
        result = rasterize_ship_svg(svg, 20, 20)
        # 全ピクセルが alpha == 255 であること
        center_alpha = result[10, 10, 3]
        assert center_alpha == 255
        # 色は暗くなっているはず (100 * 0.6 ≈ 60)
        assert result[10, 10, 0] < 100

    def test_corner_pixels_transparent(self) -> None:
        """角のピクセルが透明。"""
        result = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80)
        assert result[0, 0, 3] == 0

    def test_renders_rect_elements(self) -> None:
        """rect 要素がレンダリングされる。"""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">'
            '  <rect x="0.1" y="0.1" width="0.8" height="0.8" fill="rgb(200,100,50)"/>'
            "</svg>"
        )
        result = rasterize_ship_svg(svg, 20, 20)
        # Center should be opaque
        assert result[10, 10, 3] > 0

    def test_renders_circle_elements(self) -> None:
        """circle 要素がレンダリングされる。"""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1">'
            '  <circle cx="0.5" cy="0.5" r="0.3" fill="rgb(100,100,100)"/>'
            "</svg>"
        )
        result = rasterize_ship_svg(svg, 20, 20)
        assert result[10, 10, 3] > 0

    def test_rotation_angle_deg_zero(self) -> None:
        """angle_deg=0 は無回転と同じ結果。"""
        result_no_angle = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80)
        result_zero_angle = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80, angle_deg=0.0)
        np.testing.assert_array_almost_equal(result_no_angle, result_zero_angle)

    def test_rotation_90_returns_rotated_bbox(self) -> None:
        """90度回転で幅と高さが入れ替わる。"""
        result = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80, angle_deg=90.0)
        # viewBox 0 0 1 4, width_px=20, height_px=80
        # 90° rotation: out_w=80, out_h=20 → shape=(20, 80, 4)
        assert result.shape == (20, 80, 4)
        assert result.dtype == np.uint8

    def test_rotation_90_preserves_pixels(self) -> None:
        """90度回転後も船のピクセルが存在する。"""
        result = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80, angle_deg=90.0)
        assert (result[:, :, 3] > 0).sum() > 0

    def test_rotation_45_expands_bbox(self) -> None:
        """45度回転で出力bboxが元と異なるサイズになる。"""
        result_0 = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80)
        result_45 = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80, angle_deg=45.0)
        # 20×80 ship at 45°: out = round((20+80)*cos45) ≈ 71 for both axes
        assert result_45.shape[1] > result_0.shape[1]  # width grows (71 > 20)
        assert result_45.shape[0] != result_0.shape[0]  # height changes
        # Ship pixels should be preserved
        assert (result_45[:, :, 3] > 0).sum() > 0

    def test_no_dark_fringe_on_rotated_white_ship(self) -> None:
        """回転後のエッジに黒フリンジが出ないこと（プリマルチプライドアルファの検証）。

        透明黒キャンバス上の白い矩形を回転させると、Lanczos のストレートアルファ
        補間によってエッジ半透明ピクセルの RGB が黒に引き寄せられる（黒フリンジ）。
        プリマルチプライド処理ではこれが起きない。
        """
        # White filled rectangle covering most of the viewBox
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 5">'
            '  <rect x="0.05" y="0.05" width="0.9" height="4.9" fill="rgb(255,255,255)"/>'
            "</svg>"
        )
        result = rasterize_ship_svg(svg, 20, 100, angle_deg=30.0)
        rgba = result.astype(np.float32)
        alpha = rgba[:, :, 3]
        # Semi-transparent edge pixels: alpha in (10, 245)
        semi = (alpha > 10) & (alpha < 245)
        if not semi.any():
            return  # no semi-transparent pixels at this resolution — skip
        # For a white ship, RGB of semi-transparent pixels must be close to white.
        # With premultiplied alpha: RGB ≈ 255 before compositing.
        # With straight alpha (broken): RGB ≈ 255 * (alpha/255) → dark fringe.
        edge_rgb_mean = rgba[:, :, :3][semi].mean()
        # Require average edge RGB > 200 out of 255 (white, not dark)
        assert edge_rgb_mean > 200, (
            f"Dark fringe detected on rotated ship edges: mean RGB={edge_rgb_mean:.1f}"
        )

    def test_exclude_hull_keeps_detail_layers(self) -> None:
        """exclude_hull=True でも hull 上の detail は描画される。"""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4">'
            '  <defs>'
            '    <clipPath id="h">'
            '      <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1"/>'
            '    </clipPath>'
            '  </defs>'
            '  <polygon points="0.5,0 1,1 1,3 0.5,4 0,3 0,1" fill="rgb(120,120,120)" stroke="rgb(20,20,20)"/>'
            '  <g clip-path="url(#h)">'
            '    <circle cx="0.5" cy="2" r="0.25" fill="rgb(255,255,255)"/>'
            '  </g>'
            '</svg>'
        )

        result = rasterize_ship_svg(svg, 40, 160, exclude_hull=True)

        assert result[80, 20, 3] > 0
        assert result[80, 20, :3].mean() > 200
        assert result[120, 20, 3] == 0
