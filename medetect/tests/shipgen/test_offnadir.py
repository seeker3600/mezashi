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


def _polygon_points_by_role(svg_text: str, role: str) -> list[tuple[float, float]]:
    """Return polygon points for the first element with the requested data-role."""
    root = ET.fromstring(svg_text)
    element = root.find(f".//*[@data-role='{role}']")
    if element is None:
        raise AssertionError(f"missing polygon role: {role}")
    points_attr = element.get("points")
    if points_attr is None:
        raise AssertionError(f"polygon role has no points: {role}")
    points: list[tuple[float, float]] = []
    for pair in points_attr.split():
        x_str, y_str = pair.split(",", 1)
        points.append((float(x_str), float(y_str)))
    return points


def _visible_side_attr(svg_text: str) -> str:
    root = ET.fromstring(svg_text)
    return root.get("data-visible-side", "none")


def _struct_cx_values(svg_text: str) -> list[float]:
    """Return centre-x of each struct rect found in SVG (estimated as (x + width/2))."""
    root = ET.fromstring(svg_text)
    cx_vals: list[float] = []
    for el in root.iter():
        role = el.get("data-role", "")
        if role == "struct":
            x = el.get("x")
            w = el.get("width")
            if x is not None and w is not None:
                cx_vals.append(float(x) + float(w) / 2.0)
    return cx_vals


def _struct_cy_values(svg_text: str) -> list[float]:
    """Return centre-y of each struct rect found in SVG (estimated as (y + height/2))."""
    root = ET.fromstring(svg_text)
    cy_vals: list[float] = []
    for el in root.iter():
        role = el.get("data-role", "")
        if role == "struct":
            y = el.get("y")
            h = el.get("height")
            if y is not None and h is not None:
                cy_vals.append(float(y) + float(h) / 2.0)
    return cy_vals


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

    def test_starboard_structs_shift_left(self) -> None:
        """az=90° (starboard センサー) のとき構造物はセンサーと反対側（左）にシフトする。"""
        cx_stbd = self._struct_cx_avg(90.0)
        cx_port = self._struct_cx_avg(270.0)
        assert cx_stbd < cx_port

    def test_beam_shift_sign_symmetry(self) -> None:
        """az=90 と az=270 のシフト量は nadir 基準で逆符号かつほぼ等しい。"""
        cx_nadir = self._struct_cx_avg(0.0, offnadir=0.0)
        cx_stbd = self._struct_cx_avg(90.0)
        cx_port = self._struct_cx_avg(270.0)
        shift_stbd = cx_stbd - cx_nadir
        shift_port = cx_port - cx_nadir
        # starboard センサー → 構造物は左（センサー反対側）にシフト → 負
        # port センサー → 構造物は右（センサー反対側）にシフト → 正
        assert shift_stbd < 0 and shift_port > 0
        # 大きさはほぼ等しい（sin(90)=sin(270)=1 なので tan_theta は同じ）
        assert abs(abs(shift_stbd) - abs(shift_port)) < 0.05


class TestLengthShiftSymmetry:
    def _struct_cy_avg(self, az: float, offnadir: float = 20.0) -> float:
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(42),
            offnadir_deg=offnadir,
            sensor_az_ship_deg=az,
        )
        vals = _struct_cy_values(svg)
        return sum(vals) / len(vals) if vals else 0.5

    def test_bow_sensor_structs_shift_aft(self) -> None:
        """az=0° (bow センサー) のとき構造物はセンサー反対側の aft へシフトする。"""
        cy_bow = self._struct_cy_avg(0.0)
        cy_stern = self._struct_cy_avg(180.0)
        assert cy_bow > cy_stern

    def test_length_shift_sign_symmetry(self) -> None:
        """az=0 と az=180 のシフト量は nadir 基準で逆符号かつほぼ等しい。"""
        cy_nadir = self._struct_cy_avg(0.0, offnadir=0.0)
        cy_bow = self._struct_cy_avg(0.0)
        cy_stern = self._struct_cy_avg(180.0)
        shift_bow = cy_bow - cy_nadir
        shift_stern = cy_stern - cy_nadir
        assert shift_bow > 0 and shift_stern < 0
        assert abs(abs(shift_bow) - abs(shift_stern)) < 0.08


class TestHullProjection:
    def test_nadir_deck_matches_hull(self) -> None:
        """nadir では deck-top polygon は waterline hull と一致する。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(11),
            offnadir_deg=0.0,
            sensor_az_ship_deg=90.0,
        )
        hull = _polygon_points_by_role(svg, "hull-waterline")
        deck = _polygon_points_by_role(svg, "deck-top")
        assert deck == pytest.approx(hull)

    def test_starboard_offnadir_insets_starboard_deck_edge(self) -> None:
        """starboard 可視時は starboard 側の deck edge が内側へ入る。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(12),
            offnadir_deg=20.0,
            sensor_az_ship_deg=90.0,
        )
        hull = _polygon_points_by_role(svg, "hull-waterline")
        deck = _polygon_points_by_role(svg, "deck-top")
        mid_idx = len(hull) // 4
        assert deck[mid_idx][0] < hull[mid_idx][0]

    def test_port_offnadir_insets_port_deck_edge(self) -> None:
        """port 可視時は port 側の deck edge が内側へ入る。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(12),
            offnadir_deg=20.0,
            sensor_az_ship_deg=270.0,
        )
        hull = _polygon_points_by_role(svg, "hull-waterline")
        deck = _polygon_points_by_role(svg, "deck-top")
        mid_idx = len(hull) * 3 // 4
        assert deck[mid_idx][0] > hull[mid_idx][0]

    def test_bow_offnadir_insets_bow_deck_edge(self) -> None:
        """bow 可視時は bow 側の deck edge が aft へ入る。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(12),
            offnadir_deg=20.0,
            sensor_az_ship_deg=0.0,
        )
        hull = _polygon_points_by_role(svg, "hull-waterline")
        deck = _polygon_points_by_role(svg, "deck-top")
        assert deck[0][1] > hull[0][1]
        assert deck[-1][1] > hull[-1][1]

    def test_stern_offnadir_insets_stern_deck_edge(self) -> None:
        """stern 可視時は stern 側の deck edge が bow へ入る。"""
        svg = generate_ship_svg(
            "destroyer",
            rng=random.Random(12),
            offnadir_deg=20.0,
            sensor_az_ship_deg=180.0,
        )
        hull = _polygon_points_by_role(svg, "hull-waterline")
        deck = _polygon_points_by_role(svg, "deck-top")
        stern_starboard_idx = len(hull) // 2 - 1
        stern_port_idx = len(hull) // 2
        assert deck[stern_starboard_idx][1] < hull[stern_starboard_idx][1]
        assert deck[stern_port_idx][1] < hull[stern_port_idx][1]


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
