from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from medetect.shipgen.gen import (
    generate_ship_svg,
    generate_ships,
    get_ship_classes,
)
from medetect.shipgen.hull import build_hull_points, interpolate_hull
from medetect.shipgen.ship_class import SHIP_CLASSES, ShipColors, sample_colors

SVG_NS = "http://www.w3.org/2000/svg"


# ── Hull interpolation ───────────────────────────────────────────────────


class TestInterpolateHull:
    def test_returns_correct_length(self) -> None:
        """結果の配列長が指定した点数と一致する。"""
        hw = interpolate_hull("warship", 0.5, 0.15, 100)
        assert len(hw) == 100

    def test_values_in_range(self) -> None:
        """半幅が [0, 0.5] の範囲に収まる。"""
        hw = interpolate_hull("warship", 0.5, 0.15, 100)
        assert np.all(hw >= 0.0)
        assert np.all(hw <= 0.5)

    def test_bow_starts_narrow_for_warship(self) -> None:
        """軍艦型の艦首は幅ゼロから始まる。"""
        hw = interpolate_hull("warship", 0.8, 0.15, 100)
        assert hw[0] == pytest.approx(0.0)

    def test_sharper_bow_is_narrower(self) -> None:
        """bow_sharpness が大きいほど前方が細くなる。"""
        hw_blunt = interpolate_hull("warship", 0.0, 0.15, 100)
        hw_sharp = interpolate_hull("warship", 1.0, 0.15, 100)
        assert hw_sharp[10] < hw_blunt[10]

    def test_stern_width_applied(self) -> None:
        """指定した艦尾幅が適用される。"""
        hw = interpolate_hull("warship", 0.5, 0.30, 100)
        assert hw[-1] == pytest.approx(0.30)

    @pytest.mark.parametrize(
        "profile", ["warship", "carrier", "box", "fishing", "fishing_wide"],
    )
    def test_all_profiles_work(self, profile: str) -> None:
        """全プロファイルがエラーなく補間できる。"""
        hw = interpolate_hull(profile, 0.5, 0.15, 50)
        assert len(hw) == 50
        assert np.all(hw >= 0.0)


# ── Hull polygon ─────────────────────────────────────────────────────────


class TestBuildHullPoints:
    def test_symmetry_no_noise(self) -> None:
        """ノイズなしで船体ポリゴンが左右対称になる。"""
        hw = interpolate_hull("warship", 0.5, 0.15, 50)
        rng = random.Random(42)
        pts = build_hull_points(hw, 8.0, rng, noise_scale=0.0)

        n = len(hw)
        for i in range(n):
            rx, ry = pts[i]
            lx, ly = pts[2 * n - 1 - i]
            assert ry == pytest.approx(ly)
            assert rx + lx == pytest.approx(1.0, abs=0.01)

    def test_within_normalised_bounds(self) -> None:
        """ポリゴンの頂点が正規化座標内に収まる。"""
        hw = interpolate_hull("warship", 0.5, 0.15, 50)
        rng = random.Random(42)
        pts = build_hull_points(hw, 8.0, rng, noise_scale=0.0)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        assert min(xs) >= 0.0
        assert max(xs) <= 1.0
        assert min(ys) >= 0.0
        assert max(ys) <= 8.0 + 0.01


# ── Colour system ────────────────────────────────────────────────────────


class TestShipColors:
    def test_hull_css_format(self) -> None:
        """hull_css が有効な rgb() 文字列を返す。"""
        c = ShipColors(hull=(140, 143, 146), struct_base=(140, 143, 146))
        css = c.hull_css()
        assert css.startswith("rgb(")
        assert css.endswith(")")

    def test_struct_css_applies_offset(self) -> None:
        """struct_css が brightness_off を加算する。"""
        c = ShipColors(hull=(100, 100, 100), struct_base=(100, 100, 100))
        css = c.struct_css(brightness_off=50)
        assert "150" in css

    def test_detail_css_clamps(self) -> None:
        """detail_css が 0–255 でクランプする。"""
        c = ShipColors(hull=(250, 10, 250), struct_base=(250, 10, 250))
        light = c.detail_css(50)
        assert "255" in light  # 250+50 clamped
        dark = c.detail_css(-50)
        assert "0" in dark  # 10-50 clamped

    @pytest.mark.parametrize(
        "family", ["navy_gray", "navy_dark", "fishing_mixed", "fishing_white"],
    )
    def test_sample_colors_returns_valid(self, family: str) -> None:
        """全カラーファミリーで ShipColors を返す。"""
        colors = sample_colors(family, random.Random(42))
        assert isinstance(colors, ShipColors)
        for ch in colors.hull:
            assert 0 <= ch <= 255


# ── SVG generation ───────────────────────────────────────────────────────


class TestGenerateShipSvg:
    def test_returns_valid_xml(self) -> None:
        """生成 SVG が有効な XML である。"""
        svg = generate_ship_svg("patrol", rng=random.Random(42))
        root = ET.fromstring(svg)
        assert root.tag == f"{{{SVG_NS}}}svg"

    def test_has_viewbox(self) -> None:
        """viewBox 属性が設定されている。"""
        svg = generate_ship_svg("corvette", rng=random.Random(42))
        root = ET.fromstring(svg)
        assert "viewBox" in root.attrib

    def test_has_hull_polygon(self) -> None:
        """hull の <polygon> を含む。"""
        svg = generate_ship_svg("frigate", rng=random.Random(42))
        root = ET.fromstring(svg)
        polygons = root.findall(f"{{{SVG_NS}}}polygon")
        assert len(polygons) >= 1

    def test_has_ship_class_attr(self) -> None:
        """data-ship-class 属性が正しくセットされている。"""
        svg = generate_ship_svg("destroyer", rng=random.Random(42))
        root = ET.fromstring(svg)
        assert root.attrib["data-ship-class"] == "destroyer"

    def test_deterministic_with_seed(self) -> None:
        """同一シードで同じ SVG が生成される。"""
        svg1 = generate_ship_svg("frigate", rng=random.Random(99))
        svg2 = generate_ship_svg("frigate", rng=random.Random(99))
        assert svg1 == svg2

    @pytest.mark.parametrize("ship_class", get_ship_classes())
    def test_all_classes_generate(self, ship_class: str) -> None:
        """全艦種でエラーなく SVG を生成できる。"""
        svg = generate_ship_svg(ship_class, rng=random.Random(42))
        root = ET.fromstring(svg)
        assert root.tag == f"{{{SVG_NS}}}svg"


# ── Ship class registry ──────────────────────────────────────────────────


class TestGetShipClasses:
    def test_returns_nonempty_sorted_list(self) -> None:
        """利用可能な艦種リストが空でなくソートされている。"""
        classes = get_ship_classes()
        assert len(classes) > 0
        assert classes == sorted(classes)

    def test_matches_registry(self) -> None:
        """レジストリのキーと一致する。"""
        assert set(get_ship_classes()) == set(SHIP_CLASSES)


# ── Batch generation ─────────────────────────────────────────────────────


class TestGenerateShips:
    def test_creates_correct_number_of_files(self, tmp_path: Path) -> None:
        """指定枚数の SVG ファイルが出力される。"""
        generate_ships(
            output_dir=tmp_path,
            count=5,
            types={"patrol": 1.0},
            seed=42,
        )
        files = list(tmp_path.glob("*.svg"))
        assert len(files) == 5

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """存在しない出力ディレクトリが自動で作成される。"""
        out = tmp_path / "sub" / "dir"
        generate_ships(
            output_dir=out,
            count=1,
            types={"patrol": 1.0},
            seed=0,
        )
        assert out.is_dir()
        assert len(list(out.glob("*.svg"))) == 1

    def test_unknown_class_raises(self, tmp_path: Path) -> None:
        """不明な艦種を指定すると ValueError が発生する。"""
        with pytest.raises(ValueError, match="Unknown ship class"):
            generate_ships(
                output_dir=tmp_path,
                count=1,
                types={"nonexistent_xyz": 1.0},
            )

    def test_default_types_uses_all_classes(self, tmp_path: Path) -> None:
        """types 未指定で全クラスから均等にサンプリングされる。"""
        generate_ships(
            output_dir=tmp_path,
            count=20,
            seed=42,
        )
        files = list(tmp_path.glob("*.svg"))
        assert len(files) == 20
