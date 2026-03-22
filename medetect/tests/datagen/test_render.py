from __future__ import annotations

import numpy as np
import pytest

from medetect.datagen.render import parse_color, parse_svg_metadata, rasterize_ship_svg


class TestParseColor:
    def test_rgb_format(self) -> None:
        """rgb(r,g,b) 形式をパースする。"""
        assert parse_color("rgb(128,64,32)") == (128, 64, 32)

    def test_rgb_with_high_values(self) -> None:
        """rgb上限値をパースする。"""
        assert parse_color("rgb(255,255,255)") == (255, 255, 255)

    def test_none_returns_fallback(self) -> None:
        """未知の形式ではフォールバック色を返す。"""
        r, g, b = parse_color("unknown")
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
        """船体中央のピクセルが不透明。"""
        result = rasterize_ship_svg(self._SIMPLE_SVG, 20, 80)
        # Center of the hull should have alpha > 0
        center_alpha = result[40, 10, 3]
        assert center_alpha > 0

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
