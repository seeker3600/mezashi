"""Tests for medetect.debugging.ship_profile."""

from __future__ import annotations

import numpy as np

from medetect.debugging.ship_profile import (
    analyze_beam_profile,
    composite_rgba_on_background,
    sample_midship_beam_profile,
    summarize_profile_values,
)


class TestSampleMidshipBeamProfile:
    def test_uniform_rect_returns_uniform_profile(self) -> None:
        """一様な矩形船体では断面プロファイルも一様になる。"""
        rgba = np.zeros((120, 80, 4), dtype=np.uint8)
        rgba[30:90, 12:68, :3] = 160
        rgba[30:90, 12:68, 3] = 255

        positions, brightness, x0_px, y_px, x1_px, row_count = sample_midship_beam_profile(rgba)

        assert positions[0] == 0.0
        assert positions[-1] == 1.0
        assert np.allclose(brightness, 160.0)
        assert x0_px == 12
        assert x1_px == 67
        assert 30 <= y_px <= 89
        assert row_count > 0


class TestAnalyzeBeamProfile:
    def test_dark_outline_is_detected(self) -> None:
        """両縁が内側より暗い場合は dark outline と判定される。"""
        positions = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        brightness = np.full(100, 120.0, dtype=np.float32)
        brightness[:12] = 80.0
        brightness[-12:] = 82.0

        metrics = analyze_beam_profile(positions, brightness)

        assert metrics.has_dark_outline
        assert not metrics.has_bright_outline

    def test_bright_outline_is_detected(self) -> None:
        """両縁が内側より明るい場合は bright outline と判定される。"""
        positions = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        brightness = np.full(100, 120.0, dtype=np.float32)
        brightness[:12] = 150.0
        brightness[-12:] = 152.0

        metrics = analyze_beam_profile(positions, brightness)

        assert metrics.has_bright_outline
        assert not metrics.has_dark_outline

    def test_asymmetric_side_shading_is_not_outline(self) -> None:
        """片側だけの陰影変化は縁取り扱いしない。"""
        positions = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        brightness = np.linspace(90.0, 140.0, 100, dtype=np.float32)

        metrics = analyze_beam_profile(positions, brightness)

        assert not metrics.has_dark_outline
        assert not metrics.has_bright_outline


class TestCompositeRgbaOnBackground:
    def test_alpha_composite_blends_ship_on_background(self) -> None:
        """RGBA 船体を背景色へ合成できる。"""
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[:, :, :3] = (200, 100, 50)
        rgba[:, :, 3] = 128

        rgb = composite_rgba_on_background(rgba, bg_color=(0, 0, 0))

        assert rgb.shape == (2, 2, 3)
        assert (rgb[:, :, 0] > 90).all()


class TestSummarizeProfileValues:
    def test_rgb_profile_detects_bright_outline(self) -> None:
        """RGB プロファイルでも両縁の明縁を検出できる。"""
        values = np.full((100, 3), 120, dtype=np.uint8)
        values[:12] = (160, 160, 160)
        values[-12:] = (162, 162, 162)

        result = summarize_profile_values(values)

        assert result.has_bright_outline
        assert not result.has_dark_outline