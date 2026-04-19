from __future__ import annotations

import random

import numpy as np
import pytest

from medetect.datagen.scene import (
    _blend_rgba_layer,
    _composite_rgba,
    _darken_rgba_layer,
    _make_shadow_rgba,
    _render_ship,
    _sample_shadow_alpha,
    _shadow_alpha_for_ship,
    _shadow_blur_sigma,
    _shadow_offset_pixels,
    blend_shadow,
    blend_ship,
)


class TestBlendShip:
    def test_modifies_background(self) -> None:
        """船をブレンドすると背景が変わる。"""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        ship = np.full((10, 5, 4), 200, dtype=np.uint8)
        blend_ship(bg, ship, cx=50, cy=50, alpha_factor=1.0)
        assert bg[50, 50].sum() > 0

    def test_transparent_ship_no_change(self) -> None:
        """完全透明の船は背景を変えない。"""
        bg = np.full((100, 100, 3), 50, dtype=np.uint8)
        ship = np.zeros((10, 5, 4), dtype=np.uint8)
        original = bg.copy()
        blend_ship(bg, ship, cx=50, cy=50, alpha_factor=1.0)
        np.testing.assert_array_equal(bg, original)

    def test_clipping_at_boundary(self) -> None:
        """画像端でクリッピングされてもエラーにならない。"""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        ship = np.full((20, 10, 4), 200, dtype=np.uint8)
        blend_ship(bg, ship, cx=2, cy=2, alpha_factor=0.8)


class TestCompositeRgba:
    """_composite_rgba のテスト。"""

    def test_opaque_over_transparent(self) -> None:
        """透明背景に不透明船を重ねると船の色がそのまま出る。"""
        dst = np.zeros((20, 20, 4), dtype=np.uint8)
        src = np.full((10, 10, 4), 200, dtype=np.uint8)
        src[:, :, 3] = 255
        _composite_rgba(dst, src, 5, 5)
        assert dst[10, 10, 0] == 200
        assert dst[10, 10, 3] == 255

    def test_transparent_src_no_change(self) -> None:
        """透明な src を重ねても dst は変わらない。"""
        dst = np.full((20, 20, 4), 100, dtype=np.uint8)
        src = np.zeros((10, 10, 4), dtype=np.uint8)
        _composite_rgba(dst, src, 5, 5)
        assert dst[10, 10, 0] == 100

    def test_two_ships_gap_shows_through(self) -> None:
        """並んだ船の間の透明部分が残る。"""
        buf = np.zeros((50, 50, 4), dtype=np.uint8)
        ship_a = np.zeros((40, 10, 4), dtype=np.uint8)
        ship_a[:, :, :3] = 180
        ship_a[:, :, 3] = 255
        ship_b = np.zeros((40, 10, 4), dtype=np.uint8)
        ship_b[:, :, :3] = 160
        ship_b[:, :, 3] = 255

        _composite_rgba(buf, ship_a, 5, 5)
        _composite_rgba(buf, ship_b, 16, 5)
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
        assert bg[5, 5, 0] != 100
        assert bg[0, 0, 0] == 100

    def test_blends_without_water_tint(self) -> None:
        """water_tint が None の場合、船の色をそのまま使う。"""
        bg = np.full((10, 10, 3), 100, dtype=np.uint8)
        layer = np.zeros((10, 10, 4), dtype=np.uint8)
        layer[3:7, 3:7, :3] = 200
        layer[3:7, 3:7, 3] = 255

        _blend_rgba_layer(bg, layer, 1.0, None)
        assert bg[5, 5, 0] != 100
        assert bg[0, 0, 0] == 100


class TestShadowHelpers:
    """影レンダリング補助のテスト。"""

    def test_darken_rgba_layer_only_changes_alpha_region(self) -> None:
        """アルファがある部分だけ背景が暗くなる。"""
        bg = np.full((10, 10, 3), 120, dtype=np.uint8)
        layer = np.zeros((10, 10, 4), dtype=np.uint8)
        layer[3:7, 3:7, 3] = 255

        _darken_rgba_layer(bg, layer, 0.5)

        assert bg[5, 5, 0] < 120
        assert bg[0, 0, 0] == 120

    def test_darken_rgba_layer_respects_clip_mask(self) -> None:
        """clip_mask=False の領域は暗くならない。"""
        bg = np.full((8, 8, 3), 100, dtype=np.uint8)
        layer = np.zeros((8, 8, 4), dtype=np.uint8)
        layer[:, :, 3] = 255
        clip_mask = np.zeros((8, 8), dtype=bool)
        clip_mask[:, :4] = True

        _darken_rgba_layer(bg, layer, 0.4, clip_mask=clip_mask)

        assert bg[4, 2, 0] < 100
        assert bg[4, 6, 0] == 100

    def test_shadow_parameters_vary_with_length(self) -> None:
        """長い影設定ほど影が長く、ぼけも広がる。"""
        low_offset = _shadow_offset_pixels(beam_px=8, length_px=40, azimuth_rad=0.0, shadow_length=0.25)
        high_offset = _shadow_offset_pixels(beam_px=8, length_px=40, azimuth_rad=0.0, shadow_length=3.0)
        low_length = float(np.hypot(*low_offset))
        high_length = float(np.hypot(*high_offset))

        assert high_length > low_length
        assert _shadow_blur_sigma(8, 40, high_length) > _shadow_blur_sigma(8, 40, low_length)

    def test_sample_shadow_alpha_varies_between_tiles(self) -> None:
        """影の基準濃さは画像ごとに少しだけ変わる。"""
        rng = random.Random(123)

        values = [_sample_shadow_alpha(rng) for _ in range(6)]

        assert len({round(value, 6) for value in values}) > 1
        assert min(values) >= 0.08
        assert max(values) <= 0.11

    def test_shadow_alpha_for_ship_is_subtle_size_bias(self) -> None:
        """大型船ほど少し濃いが、差は大きくなり過ぎない。"""
        small = _shadow_alpha_for_ship(8, 40)
        medium = _shadow_alpha_for_ship(16, 88)
        large = _shadow_alpha_for_ship(28, 160)

        assert 1.0 <= small < medium < large <= 1.12
        assert large - small < 0.1

    def test_make_shadow_rgba_stays_attached_to_ship_footprint(self) -> None:
        """生成された影マスクは船体の影側から連続して伸びる。"""
        ship = np.zeros((6, 4, 4), dtype=np.uint8)
        ship[:, :, 3] = 255

        shadow = _make_shadow_rgba(
            ship,
            offset_x=8,
            offset_y=0,
            blur_sigma=0.0,
            alpha_scale=1.0,
        )

        center_row = shadow[shadow.shape[0] // 2, :, 3]
        xx = np.where(center_row > 0)[0]

        assert shadow.shape[0] > ship.shape[0]
        assert shadow.shape[1] > ship.shape[1]
        assert len(xx) > 0, "shadow must have non-zero pixels"
        # Shadow extends rightward (offset_x > 0) beyond the hull.
        hull_right = shadow.shape[1] // 2 + 2  # rough hull right edge
        assert xx[-1] > hull_right
        # Shadow pixels are contiguous (no gap).
        assert np.diff(xx).max() == 1

    def test_blend_shadow_is_noop_when_alpha_scale_zero(self) -> None:
        """alpha_factor=0 の影は背景を変えない。"""
        bg = np.full((20, 20, 3), 90, dtype=np.uint8)
        original = bg.copy()
        shadow = np.zeros((8, 8, 4), dtype=np.uint8)
        shadow[:, :, 3] = 255

        blend_shadow(bg, shadow, cx=10, cy=10, alpha_factor=0.0)

        np.testing.assert_array_equal(bg, original)

    def test_blend_shadow_allows_alpha_factor_above_one(self) -> None:
        """alpha_factor > 1 で影をより濃くできる。"""
        bg_default = np.full((20, 20, 3), 90, dtype=np.uint8)
        bg_boosted = bg_default.copy()
        shadow = np.zeros((8, 8, 4), dtype=np.uint8)
        shadow[:, :, 3] = 160

        blend_shadow(bg_default, shadow, cx=10, cy=10, alpha_factor=1.0)
        blend_shadow(bg_boosted, shadow, cx=10, cy=10, alpha_factor=1.6)

        assert bg_boosted[10, 10, 0] < bg_default[10, 10, 0]

    def test_shadow_is_one_sided(self) -> None:
        """影は太陽の反対側にのみ投影され、太陽側に暗化ピクセルがない。"""
        ship = np.zeros((12, 8, 4), dtype=np.uint8)
        ship[:, :, 3] = 255

        # offset_x=10 → shadow extends rightward, sun is on the left
        shadow = _make_shadow_rgba(
            ship,
            offset_x=10,
            offset_y=0,
            blur_sigma=0.0,
            alpha_scale=1.0,
        )

        center_y = shadow.shape[0] // 2
        center_x = shadow.shape[1] // 2
        # Ship hull occupies roughly [center_x - 4, center_x + 4] in the padded canvas.
        # Sun-facing side: columns far to the left of the hull should have zero alpha.
        sun_side = shadow[center_y, : center_x - 6, 3]
        shadow_side = shadow[center_y, center_x + 6 :, 3]

        assert int(sun_side.max()) == 0, "sun-facing side must have no shadow"
        assert int(shadow_side.max()) > 0, "shadow side must have non-zero alpha"

    def test_shadow_zero_offset_returns_empty(self) -> None:
        """offset=(0,0) で空の影配列を返す（太陽直上）。"""
        ship = np.zeros((8, 6, 4), dtype=np.uint8)
        ship[:, :, 3] = 255

        shadow = _make_shadow_rgba(
            ship,
            offset_x=0,
            offset_y=0,
            blur_sigma=1.0,
            alpha_scale=1.0,
        )

        assert shadow.shape[:2] == ship.shape[:2]
        assert int(shadow[:, :, 3].max()) == 0

    def test_shadow_hull_occlusion(self) -> None:
        """不透明な船体の直下で影アルファが抑制される。"""
        ship = np.zeros((12, 8, 4), dtype=np.uint8)
        ship[:, :, 3] = 255  # fully opaque hull

        shadow = _make_shadow_rgba(
            ship,
            offset_x=6,
            offset_y=0,
            blur_sigma=0.0,
            alpha_scale=1.0,
        )

        # The hull footprint in the padded canvas
        pad_x = abs(6) + 3  # blur_sigma=0 so ceil(0*2.5)=0
        pad_y = 3
        hull_alpha = shadow[pad_y : pad_y + 12, pad_x : pad_x + 8, 3]
        assert int(hull_alpha.max()) == 0, "shadow under opaque hull must be zero"


class TestAntiAliasedEdges:
    """スーパーサンプリング + PSF ブラーの確認。"""

    def test_alpha_edges_are_soft(self) -> None:
        """生成された船のアルファ端部に中間値が存在する。"""
        rng = random.Random(7)
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 5"'
            ' data-ship-class="destroyer" data-lb-ratio="5.0">'
            '<polygon points="0.5,0 0,5 1,5" fill="#888"/></svg>'
        )
        rgba, *_ = _render_ship(svg, 5.0, rng, 0.8, length_range=(80.0, 100.0))
        alpha = rgba[:, :, 3]
        partial = (alpha > 0) & (alpha < 255)
        assert partial.sum() > 0

    def test_interior_remains_opaque(self) -> None:
        """内部ピクセルが半透明にならない。"""
        rng = random.Random(7)
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 5"'
            ' data-ship-class="destroyer" data-lb-ratio="5.0">'
            '<polygon points="0.5,0 0,5 1,5" fill="#888"/></svg>'
        )
        rgba, *_ = _render_ship(svg, 2.0, rng, 0.5, length_range=(80.0, 100.0))
        height, width = rgba.shape[:2]
        interior = rgba[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3, 3]
        if interior.size > 0:
            assert interior.min() > 200