from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import medetect.shipgen.gen as shipgen_gen
import medetect.shipgen.hull as hull_mod
import numpy as np
import pytest

from medetect.debugging.ship_profile import render_ship_profile_metrics
from medetect.shipgen.gen import (
    generate_ship_svg,
    generate_ships,
    get_ship_classes,
)
from medetect.shipgen.hull import build_hull_points, interpolate_hull
from medetect.shipgen.ship_class import (
    SHIP_CLASSES,
    ShipColors,
    sample_colors,
    sample_hull_trait_variant,
    sample_ship_appearance_variant,
)

SVG_NS = "http://www.w3.org/2000/svg"


def _parse_rgb(css: str) -> tuple[int, int, int]:
    assert css.startswith("rgb(")
    assert css.endswith(")")
    parts = css[4:-1].split(",")
    return tuple(int(part) for part in parts)


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _chroma(rgb: tuple[int, int, int]) -> int:
    return max(rgb) - min(rgb)


def _sample_family_colors(family: str, count: int = 256) -> list[ShipColors]:
    return [sample_colors(family, random.Random(seed)) for seed in range(count)]


def _parse_points(attr: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for token in attr.split():
        x_str, y_str = token.split(",")
        points.append((float(x_str), float(y_str)))
    return points


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5


def _ship_area_from_svg(svg: str) -> float:
    root = ET.fromstring(svg)
    hull_polygon = next(root.iter(f"{{{SVG_NS}}}polygon"))
    return _polygon_area(_parse_points(hull_polygon.attrib["points"]))


def _hull_half_widths_from_svg(svg: str) -> np.ndarray:
    root = ET.fromstring(svg)
    hull_polygon = next(
        polygon
        for polygon in root.iter(f"{{{SVG_NS}}}polygon")
        if polygon.attrib.get("data-role") == "hull-waterline"
    )
    points = _parse_points(hull_polygon.attrib["points"])
    n = len(points) // 2
    half_widths = []
    for i in range(n):
        rx, ry = points[i]
        lx, ly = points[2 * n - 1 - i]
        assert ry == pytest.approx(ly)
        half_widths.append((rx - lx) * 0.5)
    return np.asarray(half_widths, dtype=np.float64)


def _struct_rect_colors(svg: str) -> list[tuple[int, int, int]]:
    root = ET.fromstring(svg)
    return [
        _parse_rgb(rect.attrib["fill"])
        for rect in root.iter(f"{{{SVG_NS}}}rect")
        if rect.attrib.get("data-role") == "struct"
    ]


def _struct_area_ratio(svg: str) -> float:
    root = ET.fromstring(svg)
    struct_area = sum(
        float(rect.attrib["width"]) * float(rect.attrib["height"])
        for rect in root.iter(f"{{{SVG_NS}}}rect")
        if rect.attrib.get("data-role") == "struct"
    )
    return struct_area / _ship_area_from_svg(svg)


def _hull_traits_from_svg(svg: str) -> set[str]:
    root = ET.fromstring(svg)
    raw = root.attrib.get("data-hull-traits", "none")
    if raw == "none":
        return set()
    return {token for token in raw.split(",") if token}


def _struct_start_positions(svg: str) -> list[float]:
    root = ET.fromstring(svg)
    lb_ratio = float(root.attrib["data-lb-ratio"])
    return [
        float(rect.attrib["y"]) / lb_ratio
        for rect in root.iter(f"{{{SVG_NS}}}rect")
        if rect.attrib.get("data-role") == "struct"
    ]


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
        "profile", ["warship", "carrier", "box", "fishing", "fishing_wide",
                     "warship_lean", "tanker"],
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


class TestHullTraitVariants:
    def test_pointed_bow_trait_narrows_forebody(self) -> None:
        """pointed bow trait は前方の船幅をさらに絞る。"""
        hw = interpolate_hull("fishing", 0.55, 0.12, 128)

        modified = hull_mod.apply_hull_trait_variant(
            hw,
            SimpleNamespace(pointed_bow=True, straight_sides=False, square_stern=False),
        )

        fore_index = len(hw) // 8
        assert modified[fore_index] < hw[fore_index] * 0.85

    def test_long_foredeck_trait_keeps_forebody_narrow_farther_aft(self) -> None:
        """long foredeck trait は pointed bow の細身領域を前甲板側へ延長する。"""
        hw = interpolate_hull("fishing", 0.55, 0.12, 128)

        pointed_only = hull_mod.apply_hull_trait_variant(
            hw,
            SimpleNamespace(
                pointed_bow=True,
                straight_sides=False,
                square_stern=False,
                long_foredeck=False,
            ),
        )
        extended = hull_mod.apply_hull_trait_variant(
            hw,
            SimpleNamespace(
                pointed_bow=True,
                straight_sides=False,
                square_stern=False,
                long_foredeck=True,
            ),
        )

        foredeck_index = int(round((len(hw) - 1) * 0.32))
        assert extended[foredeck_index] < pointed_only[foredeck_index] * 0.92

    def test_straight_sides_trait_reduces_midbody_curvature(self) -> None:
        """straight sides trait は船腹中央の幅変動を減らす。"""
        hw = interpolate_hull("fishing_wide", 0.45, 0.16, 128)

        modified = hull_mod.apply_hull_trait_variant(
            hw,
            SimpleNamespace(pointed_bow=False, straight_sides=True, square_stern=False),
        )

        mid_slice = slice(len(hw) // 4, (len(hw) * 3) // 4)
        assert float(np.std(modified[mid_slice])) < float(np.std(hw[mid_slice])) * 0.70

    def test_square_stern_trait_broadens_transom(self) -> None:
        """square stern trait は船尾の半幅を広く保つ。"""
        hw = interpolate_hull("workboat", 0.40, 0.16, 128)

        modified = hull_mod.apply_hull_trait_variant(
            hw,
            SimpleNamespace(pointed_bow=False, straight_sides=False, square_stern=True),
        )

        aft_band = modified[(len(hw) * 5) // 8:(len(hw) * 7) // 8]
        assert modified[-1] >= hw[-1] + 0.08
        assert modified[-1] >= float(np.max(aft_band)) * 0.82


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

    @pytest.mark.parametrize(
        ("family", "max_near_white", "max_mean_luminance"),
        [
            ("fishing_mixed", 18, 180.0),
            ("work_mixed", 12, 170.0),
            ("barge_dull", 10, 165.0),
        ],
    )
    def test_non_navy_families_do_not_overproduce_white_structures(
        self,
        family: str,
        max_near_white: int,
        max_mean_luminance: float,
    ) -> None:
        """非 navy 系は艦橋が白一辺倒になりすぎない。"""
        samples = [sample_colors(family, random.Random(seed)).struct_base for seed in range(64)]
        near_white = sum(
            _luminance(rgb) >= 200.0 and _chroma(rgb) <= 18
            for rgb in samples
        )
        mean_luminance = float(np.mean([_luminance(rgb) for rgb in samples]))

        assert near_white <= max_near_white
        assert mean_luminance <= max_mean_luminance

    def test_fishing_white_keeps_an_occasional_bright_superstructure_branch(self) -> None:
        """白系漁船でも極端に白い艦橋はたまに出る程度に留まる。"""
        samples = [sample_colors("fishing_white", random.Random(seed)).struct_base for seed in range(256)]
        near_white = sum(
            _luminance(rgb) >= 200.0 and _chroma(rgb) <= 18
            for rgb in samples
        )
        light_gray = sum(
            182.0 <= _luminance(rgb) < 200.0 and _chroma(rgb) <= 18
            for rgb in samples
        )
        mean_luminance = float(np.mean([_luminance(rgb) for rgb in samples]))

        assert near_white >= 16
        assert near_white <= 96
        assert light_gray >= 72
        assert 184.0 <= mean_luminance <= 198.0

    @pytest.mark.parametrize(
        ("family", "min_muted", "min_dark_muted", "min_light_muted"),
        [
            ("fishing_mixed", 116, 28, 72),
            ("work_mixed", 128, 80, 0),
            ("barge_dull", 160, 150, 0),
        ],
    )
    def test_civilian_hulls_include_low_saturation_variants(
        self,
        family: str,
        min_muted: int,
        min_dark_muted: int,
        min_light_muted: int,
    ) -> None:
        """民生系 hull は低彩度の暗色・明色バリエーションを十分に含む。"""
        samples = _sample_family_colors(family)
        hulls = [sample.hull for sample in samples]

        muted = sum(_chroma(rgb) <= 30 for rgb in hulls)
        dark_muted = sum(
            _luminance(rgb) <= 95.0 and _chroma(rgb) <= 30
            for rgb in hulls
        )
        light_muted = sum(
            _luminance(rgb) >= 185.0 and _chroma(rgb) <= 24
            for rgb in hulls
        )

        assert muted >= min_muted
        assert dark_muted >= min_dark_muted
        assert light_muted >= min_light_muted

    @pytest.mark.parametrize(
        ("family", "max_large_gap", "max_dark_hull_bright_struct"),
        [
            ("fishing_mixed", 14, 4),
            ("work_mixed", 6, 0),
            ("barge_dull", 4, 0),
        ],
    )
    def test_non_white_civilian_superstructures_do_not_float_far_above_hull_brightness(
        self,
        family: str,
        max_large_gap: int,
        max_dark_hull_bright_struct: int,
    ) -> None:
        """非 white 系 civilian は艦橋だけが不自然に明るく浮きすぎない。"""
        samples = _sample_family_colors(family)
        gaps = [
            _luminance(sample.struct_base) - _luminance(sample.hull)
            for sample in samples
        ]
        dark_hull_bright_struct = sum(
            _luminance(sample.hull) <= 105.0
            and _luminance(sample.struct_base) >= 165.0
            for sample in samples
        )

        assert sum(gap >= 60.0 for gap in gaps) <= max_large_gap
        assert dark_hull_bright_struct <= max_dark_hull_bright_struct

    def test_work_mixed_superstructures_do_not_drop_into_near_black_band(self) -> None:
        """work_mixed の構造物は near-black に落ち込まない。"""
        dark = 0

        for seed in range(256):
            colors = sample_colors("work_mixed", random.Random(seed))
            struct_fill = _parse_rgb(colors.struct_css(brightness_off=28, rng=random.Random(seed)))
            if _luminance(struct_fill) <= 76.0 and _chroma(struct_fill) <= 20:
                dark += 1

        assert dark == 0


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

    def test_has_clippath_for_deck_scatter(self) -> None:
        """甲板散布用の <clipPath id="h"> が <defs> 内に存在する。"""
        svg = generate_ship_svg("frigate", rng=random.Random(42))
        root = ET.fromstring(svg)
        defs = root.find(f"{{{SVG_NS}}}defs")
        assert defs is not None
        clip = defs.find(f"{{{SVG_NS}}}clipPath")
        assert clip is not None
        assert clip.attrib.get("id") == "h"

    def test_deck_scatter_group_present(self) -> None:
        """clip-path 属性を持つ <g> 要素（散布図形グループ）が存在する。"""
        svg = generate_ship_svg("frigate", rng=random.Random(42))
        root = ET.fromstring(svg)
        groups = [e for e in root if e.tag == f"{{{SVG_NS}}}g" and "clip-path" in e.attrib]
        assert len(groups) >= 1

    def test_deck_scatter_group_has_shapes(self) -> None:
        """散布グループ内に少なくとも1つの図形要素がある。"""
        svg = generate_ship_svg("frigate", rng=random.Random(42))
        root = ET.fromstring(svg)
        groups = [e for e in root if e.tag == f"{{{SVG_NS}}}g" and "clip-path" in e.attrib]
        assert len(groups) >= 1
        total_children = sum(len(list(g)) for g in groups)
        assert total_children > 0

    def test_deck_scatter_density_zero_is_empty(self) -> None:
        """deck_scatter_density=0 のとき散布グループが空になる。"""
        svg = generate_ship_svg("frigate", rng=random.Random(42), deck_scatter_density=0)
        root = ET.fromstring(svg)
        # Only count children in the scatter group (id="scatter"), not hull-effect groups
        scatter_groups = [
            e for e in root
            if e.tag == f"{{{SVG_NS}}}g" and e.get("id") == "scatter"
        ]
        total_children = sum(len(list(g)) for g in scatter_groups)
        assert total_children == 0

    def test_deck_scatter_density_high_more_shapes(self) -> None:
        """density が高いほど散布図形が増える（同シード比較）。"""
        svg_lo = generate_ship_svg("frigate", rng=random.Random(7), deck_scatter_density=1.0)
        svg_hi = generate_ship_svg("frigate", rng=random.Random(7), deck_scatter_density=10.0)
        def _count(svg: str) -> int:
            root = ET.fromstring(svg)
            return sum(
                len(list(g))
                for g in root
                if g.tag == f"{{{SVG_NS}}}g" and "clip-path" in g.attrib
            )
        assert _count(svg_hi) > _count(svg_lo)

    def test_deck_scatter_fills_are_never_void_like(self) -> None:
        """散乱図形の fill/stroke は暗いハルでも穴のように見える黒にならない。

        正オフセットのみ使用するため、いかなるハル色でも near-black は生成されない。
        """
        import re
        _RGB_RE = re.compile(r"rgb\((\d+),(\d+),(\d+)\)")

        def _is_void_like(r: int, g: int, b: int) -> bool:
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            return lum <= 28.0 and max(r, g, b) <= 40

        # Cover dark-hull classes across many seeds and high scatter density.
        dark_classes = ["frigate", "destroyer", "corvette"]
        for ship_class in dark_classes:
            for seed in range(30):
                svg = generate_ship_svg(
                    ship_class,
                    rng=random.Random(seed),
                    deck_scatter_density=10.0,
                )
                root = ET.fromstring(svg)
                scatter_groups = [
                    e for e in root
                    if e.tag == f"{{{SVG_NS}}}g" and e.get("id") == "scatter"
                ]
                for group in scatter_groups:
                    for elem in group:
                        for attr in ("fill", "stroke"):
                            val = elem.get(attr, "")
                            m = _RGB_RE.match(val)
                            if not m:
                                continue
                            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
                            assert not _is_void_like(r, g, b), (
                                f"{ship_class} seed={seed}: void-like {attr} "
                                f"rgb({r},{g},{b}) in scatter"
                            )

    @pytest.mark.parametrize("ship_class", get_ship_classes(include_debug=True))
    def test_all_classes_generate(self, ship_class: str) -> None:
        """全艦種でエラーなく SVG を生成できる。"""
        svg = generate_ship_svg(ship_class, rng=random.Random(42))
        root = ET.fromstring(svg)
        assert root.tag == f"{{{SVG_NS}}}svg"

    def test_debug_rect_generates_single_hull_polygon(self) -> None:
        """debug_rect は単色の長方形 hull だけを出力する。"""
        svg = generate_ship_svg("debug_rect", rng=random.Random(42))
        root = ET.fromstring(svg)

        polygons = root.findall(f"{{{SVG_NS}}}polygon")
        assert len(polygons) == 1
        assert len(polygons[0].attrib["points"].split()) == 4
        assert polygons[0].attrib["fill"] in {
            "rgb(220,48,48)",
            "rgb(48,180,72)",
            "rgb(54,104,224)",
            "rgb(236,208,48)",
        }
        assert not root.findall(f".//{{{SVG_NS}}}rect")
        assert not root.findall(f".//{{{SVG_NS}}}circle")
        assert not root.findall(f".//{{{SVG_NS}}}line")

    def test_debug_rect_profile_is_symmetric(self) -> None:
        """debug_rect の断面は単純で左右対称な矩形になる。"""
        metrics = render_ship_profile_metrics("debug_rect", seed=42, bg_color=(255, 255, 255))

        assert not metrics.has_dark_outline
        assert abs(metrics.left_edge_delta - metrics.right_edge_delta) < 1.0

    def test_hull_has_multiple_clipped_groups(self) -> None:
        """船体エフェクト用の clip-path グループが複数存在する。"""
        svg = generate_ship_svg("destroyer", rng=random.Random(42))
        root = ET.fromstring(svg)
        clipped = [e for e in root if e.tag == f"{{{SVG_NS}}}g" and "clip-path" in e.attrib]
        # At minimum: primary lighting, secondary lighting, mottling,
        # bow/stern shading, soft side asymmetry, panels, wear, scatter
        assert len(clipped) >= 7


class TestSmallShipRareVariants:
    def test_variant_flags_stay_sparse_and_independent(self) -> None:
        """小型船レアバリアントの2条件は独立に低頻度で発生する。"""
        ship_class = SHIP_CLASSES["fishing_longliner"]
        variants = [
            sample_ship_appearance_variant(ship_class, random.Random(seed))
            for seed in range(512)
        ]

        oversized = sum(variant.oversized_struct for variant in variants)
        bright = sum(variant.bright_white_struct for variant in variants)
        oversized_only = sum(
            variant.oversized_struct and not variant.bright_white_struct
            for variant in variants
        )
        bright_only = sum(
            variant.bright_white_struct and not variant.oversized_struct
            for variant in variants
        )

        assert 4 <= oversized <= 20
        assert 4 <= bright <= 20
        assert oversized_only >= 1
        assert bright_only >= 1

    @pytest.mark.parametrize(
        ("ship_class", "min_long_foredeck", "max_long_foredeck"),
        [
            ("fishing_longliner", 20, 240),
            ("fishing_purse_seiner", 20, 240),
            ("workboat", 20, 240),
        ],
    )
    def test_targeted_small_civilian_classes_include_long_foredeck_variants(
        self,
        ship_class: str,
        min_long_foredeck: int,
        max_long_foredeck: int,
    ) -> None:
        """対象小型民間船では long foredeck バリアントが実用頻度で出る。"""
        variants = [
            sample_hull_trait_variant(SHIP_CLASSES[ship_class], random.Random(seed))
            for seed in range(512)
        ]

        long_foredeck = sum(getattr(variant, "long_foredeck", False) for variant in variants)

        assert min_long_foredeck <= long_foredeck <= max_long_foredeck

    def test_long_foredeck_variant_pushes_superstructure_aft_on_longliner(self) -> None:
        """long foredeck variant は主艦橋の開始位置を後方へ寄せる。"""
        trait_starts: list[float] = []
        plain_starts: list[float] = []

        for seed in range(512):
            svg = generate_ship_svg("fishing_longliner", rng=random.Random(seed))
            starts = _struct_start_positions(svg)
            if not starts:
                continue
            if "long_foredeck" in _hull_traits_from_svg(svg):
                trait_starts.append(min(starts))
            else:
                plain_starts.append(min(starts))

        assert len(trait_starts) >= 12
        assert float(np.mean(trait_starts)) >= float(np.mean(plain_starts)) + 0.05

    @pytest.mark.parametrize(
        "ship_class",
        ["fishing_longliner", "fishing_purse_seiner", "workboat"],
    )
    def test_selected_civilian_classes_do_not_flag_low_contrast_struct_variants(
        self,
        ship_class: str,
    ) -> None:
        """対象 civilian class では low contrast 構造物バリアントを使わない。"""
        variants = [
            sample_ship_appearance_variant(SHIP_CLASSES[ship_class], random.Random(seed))
            for seed in range(512)
        ]

        subdued = sum(bool(getattr(variant, "low_contrast_struct", False)) for variant in variants)

        assert subdued == 0

    def test_small_fishing_white_superstructures_glow_only_rarely(self) -> None:
        """小型の白系漁船で白く光る構造物はたまにしか出ない。"""
        bright = 0
        for seed in range(384):
            svg = generate_ship_svg("fishing_longliner", rng=random.Random(seed))
            if any(
                _luminance(rgb) >= 220.0 and _chroma(rgb) <= 14
                for rgb in _struct_rect_colors(svg)
            ):
                bright += 1

        assert 3 <= bright <= 16

    @pytest.mark.parametrize(
        "ship_class",
        ["fishing_longliner", "fishing_purse_seiner", "workboat"],
    )
    def test_targeted_small_civilian_superstructures_do_not_go_near_black(
        self,
        ship_class: str,
    ) -> None:
        """対象小型民間船で near-black の艦橋は生成されない。"""
        dark = 0
        for seed in range(384):
            svg = generate_ship_svg(ship_class, rng=random.Random(seed))
            if any(
                _luminance(rgb) <= 76.0 and _chroma(rgb) <= 20
                for rgb in _struct_rect_colors(svg)
            ):
                dark += 1

        assert dark == 0

    def test_small_ship_oversized_superstructures_stay_rare(self) -> None:
        """小型船で構造物が船面積の半分超えになるのは稀に留まる。"""
        oversized = 0
        for seed in range(384):
            svg = generate_ship_svg("pilot_boat", rng=random.Random(seed))
            if _struct_area_ratio(svg) > 0.5:
                oversized += 1

        assert 3 <= oversized <= 12

    def test_hull_mottling_adds_polygons(self) -> None:
        """船体ムラ処理で半透明ポリゴンが追加される。"""
        svg = generate_ship_svg("frigate", rng=random.Random(42))
        # Mottling polygons use rgba fills
        assert "rgba(" in svg

    def test_bow_stern_shading_present(self) -> None:
        """船首・船尾の陰影が存在する。"""
        svg = generate_ship_svg("destroyer", rng=random.Random(42))
        root = ET.fromstring(svg)
        # Bow/stern shading adds multiple rgba(0,0,0,...) polygons
        clipped_groups = [e for e in root if e.tag == f"{{{SVG_NS}}}g" and "clip-path" in e.attrib]
        rgba_polygons = 0
        for g in clipped_groups:
            for child in g:
                if child.tag == f"{{{SVG_NS}}}polygon":
                    fill = child.get("fill", "")
                    if "rgba(" in fill:
                        rgba_polygons += 1
        assert rgba_polygons >= 5  # at minimum several overlay patches

    def test_struct_self_shadow_present(self) -> None:
        """上部構造物の片側に自己影がある。"""
        svg = generate_ship_svg("destroyer", rng=random.Random(42))
        root = ET.fromstring(svg)
        # Self-shadow rects are now solid rgb(...) fills derived from the
        # structure colour — they are NOT pure black and do NOT use rgba().
        rects = root.findall(f"{{{SVG_NS}}}rect")
        # At least one direct-child <rect> should exist (struct or shadow)
        assert len(rects) >= 1
        # No shadow or struct rect should use pure-black rgba fill
        black_rgba_rects = [r for r in rects if r.get("fill", "").startswith("rgba(0,0,0")]
        assert len(black_rgba_rects) == 0, "shadows should not use pure-black rgba"

    def test_struct_rects_are_tagged_for_visual_qc(self) -> None:
        """艦橋の明部と影部を SVG 上で識別できる。"""
        svg = generate_ship_svg("tug_harbor", rng=random.Random(42))
        root = ET.fromstring(svg)
        struct_rects = [
            rect for rect in root.findall(f"{{{SVG_NS}}}rect")
            if rect.get("data-role") == "struct"
        ]
        shadow_rects = [
            rect for rect in root.findall(f"{{{SVG_NS}}}rect")
            if rect.get("data-role") == "struct-shadow"
        ]

        assert struct_rects
        assert shadow_rects

    def test_forced_perimeter_trim_adds_trim_polygons(self) -> None:
        """全周 trim を強制すると hull trim ポリゴンが出力される。"""
        svg = generate_ship_svg(
            "fishing_longliner",
            rng=random.Random(42),
            trim_mode="perimeter",
        )
        root = ET.fromstring(svg)

        trim_polygons = [
            polygon for polygon in root.findall(f"{{{SVG_NS}}}polygon")
            if polygon.get("data-role") == "hull-trim"
        ]

        assert root.attrib["data-trim-mode"] == "perimeter"
        assert root.attrib["data-visible-side"] == "none"
        assert trim_polygons

    def test_forced_bow_trim_excludes_perimeter_trim(self) -> None:
        """船首 trim を強制した場合は全周 trim を同時に出さない。"""
        svg = generate_ship_svg(
            "fishing_purse_seiner",
            rng=random.Random(42),
            trim_mode="bow",
        )
        root = ET.fromstring(svg)

        bow_polygons = [
            polygon for polygon in root.findall(f"{{{SVG_NS}}}polygon")
            if polygon.get("data-role") == "bow-trim"
        ]
        perimeter_polygons = [
            polygon for polygon in root.findall(f"{{{SVG_NS}}}polygon")
            if polygon.get("data-role") == "hull-trim"
        ]

        assert root.attrib["data-trim-mode"] == "bow"
        assert bow_polygons
        assert not perimeter_polygons

    def test_forced_visible_side_tags_side_trim(self) -> None:
        """側面色を強制すると片側 trim と side metadata を残す。"""
        svg = generate_ship_svg(
            "tug_harbor",
            rng=random.Random(42),
            trim_mode="none",
            offnadir_deg=20.0,
            sensor_az_ship_deg=90.0,
        )
        root = ET.fromstring(svg)

        side_polygons = [
            polygon for polygon in root.findall(f"{{{SVG_NS}}}polygon")
            if polygon.get("data-role") == "side-trim"
        ]

        assert root.attrib["data-trim-mode"] == "none"
        assert root.attrib["data-visible-side"] == "starboard"
        assert side_polygons

    @pytest.mark.parametrize(
        "ship_class",
        ["fishing_longliner", "fishing_purse_seiner", "workboat"],
    )
    def test_selected_civilian_classes_emit_combined_and_partial_hull_traits(
        self,
        ship_class: str,
    ) -> None:
        """対象 civilian class では hull trait の複合形と部分差分が両方出る。"""
        trait_values = []
        for seed in range(384):
            svg = generate_ship_svg(ship_class, rng=random.Random(seed), hull_noise=0.0)
            root = ET.fromstring(svg)
            trait_values.append(root.attrib["data-hull-traits"])

        combined_value = "pointed_bow,long_foredeck,straight_sides,square_stern"
        combined = sum(value == combined_value for value in trait_values)
        partial = sum(value not in {"none", combined_value} for value in trait_values)

        assert 16 <= combined <= 128
        assert partial >= 24

    def test_non_target_classes_keep_hull_traits_disabled(self) -> None:
        """非対象 class では hull trait metadata が none のまま。"""
        svg = generate_ship_svg("destroyer", rng=random.Random(42), hull_noise=0.0)
        root = ET.fromstring(svg)

        assert root.attrib["data-hull-traits"] == "none"

    def test_generate_ship_svg_uses_sampled_hull_traits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate_ship_svg は sampled hull traits を hull geometry に反映する。"""
        monkeypatch.setattr(
            shipgen_gen,
            "sample_hull_trait_variant",
            lambda _cls, _rng: SimpleNamespace(
                pointed_bow=False,
                long_foredeck=False,
                straight_sides=False,
                square_stern=False,
            ),
            raising=False,
        )

        base_svg = generate_ship_svg("workboat", rng=random.Random(42), hull_noise=0.0)

        monkeypatch.setattr(
            shipgen_gen,
            "sample_hull_trait_variant",
            lambda _cls, _rng: SimpleNamespace(
                pointed_bow=True,
                long_foredeck=True,
                straight_sides=True,
                square_stern=True,
            ),
            raising=False,
        )

        variant_svg = generate_ship_svg("workboat", rng=random.Random(42), hull_noise=0.0)

        base_hw = _hull_half_widths_from_svg(base_svg)
        variant_hw = _hull_half_widths_from_svg(variant_svg)
        fore_index = len(base_hw) // 8
        mid_slice = slice((len(base_hw) * 9) // 20, (len(base_hw) * 4) // 5)

        assert variant_hw[fore_index] < base_hw[fore_index] * 0.90
        assert float(np.std(variant_hw[mid_slice])) < float(np.std(base_hw[mid_slice])) * 0.75
        assert variant_hw[-1] >= base_hw[-1] + 0.08

    def test_rendered_profiles_do_not_show_bilateral_outline(self) -> None:
        """描画後の船幅断面で両縁だけが同方向に強調されない。"""
        cases = [
            ("amphib_assault", 42),
            ("barge", 42),
            ("barge_deck", 42),
            ("carrier", 42),
            ("corvette", 42),
            ("destroyer", 42),
            ("destroyer_stealth", 42),
            ("fishing_longliner", 42),
            ("fishing_purse_seiner", 42),
            ("tug_harbor", 42),
        ]
        offenders: list[tuple[str, float, float]] = []

        for ship_class, seed in cases:
            metrics = render_ship_profile_metrics(ship_class, seed=seed)
            if metrics.has_dark_outline or metrics.has_bright_outline:
                offenders.append(
                    (
                        ship_class,
                        round(metrics.left_edge_delta, 2),
                        round(metrics.right_edge_delta, 2),
                    )
                )

        assert not offenders, f"bilateral outline detected: {offenders}"

    def test_rendered_profiles_keep_lighting_variety_without_outline(self) -> None:
        """複数シードでフラット・暗側・明側の断面が混在する。"""
        patterns: set[str] = set()
        for seed in range(40):
            metrics = render_ship_profile_metrics("frigate", seed=seed)
            assert not metrics.has_dark_outline
            assert not metrics.has_bright_outline

            left = metrics.left_edge_delta
            right = metrics.right_edge_delta
            if abs(left) < 4.0 and abs(right) < 4.0:
                patterns.add("flat")
            if min(left, right) < -5.0:
                patterns.add("shadowed_side")
            if max(left, right) > 5.0:
                patterns.add("lit_side")
            if len(patterns) == 3:
                break

        assert len(patterns) >= 3, f"断面バリエーションが不足: {patterns}"


# ── Ship class registry ──────────────────────────────────────────────────


class TestGetShipClasses:
    def test_returns_nonempty_sorted_list(self) -> None:
        """利用可能な艦種リストが空でなくソートされている。"""
        classes = get_ship_classes()
        assert len(classes) > 0
        assert classes == sorted(classes)
        assert "debug_rect" not in classes

    def test_include_debug_matches_registry(self) -> None:
        """debug class を含めるとレジストリのキーと一致する。"""
        assert set(get_ship_classes(include_debug=True)) == set(SHIP_CLASSES)


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

    def test_default_types_use_public_classes_only(self, tmp_path: Path) -> None:
        """types 未指定では debug class を除いた公開クラスだけを使う。"""
        generate_ships(
            output_dir=tmp_path,
            count=20,
            seed=42,
        )
        files = list(tmp_path.glob("*.svg"))
        assert len(files) == 20
        assert not any(path.name.startswith("debug_rect_") for path in files)

    def test_explicit_debug_rect_selection_is_allowed(self, tmp_path: Path) -> None:
        """debug_rect は明示指定したときだけ生成できる。"""
        generate_ships(
            output_dir=tmp_path,
            count=3,
            types={"debug_rect": 1.0},
            seed=42,
        )

        files = sorted(tmp_path.glob("*.svg"))
        assert len(files) == 3
        assert all(path.name.startswith("debug_rect_") for path in files)
        for path in files:
            root = ET.fromstring(path.read_text(encoding="utf-8"))
            assert root.attrib["data-ship-class"] == "debug_rect"

    def test_filetype_png_creates_png_files(self, tmp_path: Path) -> None:
        """filetype='png' で PNG ファイルが生成される。"""
        generate_ships(
            output_dir=tmp_path,
            count=3,
            types={"patrol": 1.0},
            seed=42,
            filetype="png",
        )
        files = list(tmp_path.glob("*.png"))
        assert len(files) == 3
        assert not list(tmp_path.glob("*.svg"))
        # Verify valid PNG header (magic bytes)
        for f in files:
            assert f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
