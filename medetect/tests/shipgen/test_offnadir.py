"""off-nadir 視点シミュレーションのテスト。

船のSVGに side_component / beam_shift が正しく反映されることを確認する。
"""
from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET

import pytest

from medetect.shipgen.gen import generate_ship_svg
from medetect.shipgen.ship_class import (
    _HULL_DEPTH_FRAC,
    _SIDE_COMPONENT_EPS,
    _sample_hull_trim_style,
)

SVG_NS = "http://www.w3.org/2000/svg"


def _side_trim_elements(svg_text: str) -> list[ET.Element]:
    """Return all SVG elements with data-role='side-trim'."""
    root = ET.fromstring(svg_text)
    return root.findall(f".//*[@data-role='side-trim']")


def _visible_side_attr(svg_text: str) -> str:
    root = ET.fromstring(svg_text)
    return root.get("data-visible-side", "none")


def _struct_cx_values(svg_text: str) -> list[float]:
    """Return centre-x of each struct rect found in SVG (estimated as (x + width/2))."""
    root = ET.fromstring(svg_text)
    cx_vals: list[float] = []
    for el in root.iter():
        role = el.get("data-role", "")
        if "struct" in role:
            x = el.get("x")
            w = el.get("width")
            if x is not None and w is not None:
                cx_vals.append(float(x) + float(w) / 2.0)
    return cx_vals


class TestSideComponentDirection:
    def test_starboard_visible_at_az90(self) -> None:
        """az=90° (starboard abeam) → starboard 側面バンドが存在する。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(0),
            offnadir_deg=15.0,
            sensor_az_ship_deg=90.0,
        )
        assert _visible_side_attr(svg) == "starboard"
        assert len(_side_trim_elements(svg)) > 0

    def test_port_visible_at_az270(self) -> None:
        """az=270° (port abeam) → port 側面バンドが存在する。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(0),
            offnadir_deg=15.0,
            sensor_az_ship_deg=270.0,
        )
        assert _visible_side_attr(svg) == "port"
        assert len(_side_trim_elements(svg)) > 0

    def test_no_side_trim_at_nadir(self) -> None:
        """offnadir_deg=0 → 側面バンドなし。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(0),
            offnadir_deg=0.0,
            sensor_az_ship_deg=90.0,
        )
        assert _visible_side_attr(svg) == "none"
        assert len(_side_trim_elements(svg)) == 0

    def test_no_side_trim_bow_on(self) -> None:
        """az=0° (bow-on) → side_component ≈ 0 → 側面バンドなし。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(0),
            offnadir_deg=20.0,
            sensor_az_ship_deg=0.0,
        )
        assert _visible_side_attr(svg) == "none"
        assert len(_side_trim_elements(svg)) == 0

    def test_no_side_trim_stern_on(self) -> None:
        """az=180° (stern-on) → side_component ≈ 0 → 側面バンドなし。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(0),
            offnadir_deg=20.0,
            sensor_az_ship_deg=180.0,
        )
        assert _visible_side_attr(svg) == "none"
        assert len(_side_trim_elements(svg)) == 0


class TestSideWidthMonotonicity:
    def test_larger_angle_wider_band(self) -> None:
        """offnadir が大きいほど side_width は大きい (tan 単調増加)。"""
        hull = (120, 130, 140)
        rng = random.Random(1)
        style_narrow = _sample_hull_trim_style(
            "navy_gray",
            hull,
            rng,
            side_component=math.tan(math.radians(5.0)),
        )
        rng = random.Random(1)
        style_wide = _sample_hull_trim_style(
            "navy_gray",
            hull,
            rng,
            side_component=math.tan(math.radians(30.0)),
        )
        assert style_wide.side_width > style_narrow.side_width

    def test_side_width_formula(self) -> None:
        """side_width = |side_component| * _HULL_DEPTH_FRAC を確認する。"""
        hull = (120, 130, 140)
        side_component = 0.3
        style = _sample_hull_trim_style(
            "navy_gray",
            hull,
            random.Random(2),
            side_component=side_component,
        )
        expected = abs(side_component) * _HULL_DEPTH_FRAC
        assert style.side_width == pytest.approx(expected)


class TestBeamShiftSymmetry:
    def _struct_cx_avg(self, az: float, offnadir: float = 20.0) -> float:
        svg = generate_ship_svg(
            "supply",
            rng=random.Random(42),
            offnadir_deg=offnadir,
            sensor_az_ship_deg=az,
        )
        vals = _struct_cx_values(svg)
        return sum(vals) / len(vals) if vals else 0.5

    def test_starboard_structs_shift_right(self) -> None:
        """az=90° のとき構造物は az=270° のときより右にシフトする。"""
        cx_stbd = self._struct_cx_avg(90.0)
        cx_port = self._struct_cx_avg(270.0)
        assert cx_stbd > cx_port

    def test_beam_shift_sign_symmetry(self) -> None:
        """az=90 と az=270 のシフト量は nadir 基準で逆符号かつほぼ等しい。"""
        cx_nadir = self._struct_cx_avg(0.0, offnadir=0.0)
        cx_stbd = self._struct_cx_avg(90.0)
        cx_port = self._struct_cx_avg(270.0)
        shift_stbd = cx_stbd - cx_nadir
        shift_port = cx_port - cx_nadir
        # 方向が逆
        assert shift_stbd > 0 and shift_port < 0
        # 大きさはほぼ等しい（sin(90)=sin(270)=1 なので tan_theta は同じ）
        assert abs(abs(shift_stbd) - abs(shift_port)) < 0.05


class TestSideComponentEps:
    def test_tiny_component_is_none(self) -> None:
        """side_component が _EPS 未満のときは 'none'。"""
        hull = (120, 130, 140)
        style = _sample_hull_trim_style(
            "navy_gray",
            hull,
            random.Random(3),
            side_component=_SIDE_COMPONENT_EPS * 0.5,
        )
        assert style.visible_side == "none"
        assert style.side_width == 0.0
