from __future__ import annotations

import pathlib
import random
import xml.etree.ElementTree as ET

import pytest

import medetect.datagen.ship as ship_mod
import medetect.shipgen.gen as shipgen_gen

from medetect.datagen.ship import (
    MIN_SHIP_BEAM_PX,
    _LB_OUTER_BAND_MULTIPLIER,
    _MAX_REASONABLE_LB_RATIO_MULTIPLIER,
    _pick_svg,
    _SvgMeta,
    _load_svg_metas,
    _min_reasonable_lb_ratio,
    _natural_lb_ratio,
    _scale_ship_pixel_size,
    _ship_class_id,
    _size_class_names,
    _svg_lb_weight,
    compute_ship_pixel_size,
)


class TestComputeShipPixelSize:
    def test_destroyer_at_10m(self) -> None:
        """10 m/px 解像度での駆逐艦のピクセルサイズ。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "destroyer", lb_ratio=8.0, resolution_m=10.0, rng=rng,
        )
        assert 10 <= length_px <= 25
        assert beam_px >= MIN_SHIP_BEAM_PX

    def test_fishing_trawler_at_2m(self) -> None:
        """2 m/px 解像度での漁船のピクセルサイズ。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "fishing_trawler", lb_ratio=5.0, resolution_m=2.0, rng=rng,
        )
        assert 5 <= length_px <= 30
        assert beam_px >= MIN_SHIP_BEAM_PX

    def test_minimum_pixel_size(self) -> None:
        """最小ピクセルサイズが保証される。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "fishing_trawler", lb_ratio=5.0, resolution_m=100.0, rng=rng,
        )
        assert beam_px >= MIN_SHIP_BEAM_PX
        assert length_px >= 3

    def test_unknown_class_uses_default(self) -> None:
        """未知のクラスでもデフォルトサイズで動作する。"""
        rng = random.Random(42)
        beam_px, length_px = compute_ship_pixel_size(
            "unknown_vessel", lb_ratio=6.0, resolution_m=5.0, rng=rng,
        )
        assert beam_px >= MIN_SHIP_BEAM_PX
        assert length_px >= 3

    def test_length_dependent_beam_floor_widens_slender_small_ship(self) -> None:
        """15m 級の細すぎる船は長さ依存の下限で補正される。"""
        rng = random.Random(0)
        beam_px, length_px = compute_ship_pixel_size(
            "fishing_trawler",
            lb_ratio=12.0,
            resolution_m=0.5,
            rng=rng,
            length_range=(15.0, 15.0),
        )
        assert beam_px == 5
        assert length_px == 30

    def test_natural_lb_ratio_keeps_original_beam(self) -> None:
        """自然な L/B 比の船は不要に太らせない。"""
        rng = random.Random(0)
        beam_px, length_px = compute_ship_pixel_size(
            "patrol",
            lb_ratio=4.0,
            resolution_m=0.5,
            rng=rng,
            length_range=(60.0, 60.0),
        )
        assert beam_px == 30
        assert length_px == 120


class TestScaleShipPixelSize:
    def test_beam_floor_applies_after_scaling(self) -> None:
        """クラスタ再スケール後も beam の共有下限を維持する。"""
        assert _scale_ship_pixel_size(2, 18, 0.9) == (MIN_SHIP_BEAM_PX, 16)

    def test_length_range_clamps_upper(self) -> None:
        """length_range の上限が適用される。"""
        rng = random.Random(0)
        results = [
            compute_ship_pixel_size(
                "destroyer", lb_ratio=8.0, resolution_m=1.0,
                rng=rng, length_range=(10.0, 50.0),
            )
            for _ in range(20)
        ]
        for _beam, length in results:
            assert length <= 52

    def test_length_range_clamps_lower(self) -> None:
        """length_range の下限が適用される。"""
        rng = random.Random(0)
        results = [
            compute_ship_pixel_size(
                "fishing_trawler", lb_ratio=5.0, resolution_m=1.0,
                rng=rng, length_range=(80.0, 200.0),
            )
            for _ in range(20)
        ]
        for _beam, length in results:
            assert length >= 78

    def test_length_range_none_uses_class_range(self) -> None:
        """length_range=None のとき制約なし（既存の動作を維持）。"""
        rng = random.Random(42)
        _beam_px, length_px = compute_ship_pixel_size(
            "destroyer", lb_ratio=8.0, resolution_m=10.0, rng=rng, length_range=None,
        )
        assert length_px >= 3


class TestPickSvgCaching:
    def test_onthefly_pool_keeps_multiple_variants(self, monkeypatch) -> None:
        """同一キーのオンザフライ生成でも複数の見た目を維持する。"""
        ship_mod._SHIPGEN_VARIANT_CALLS.clear()
        ship_mod._generate_ship_svg_variant.cache_clear()
        ship_mod._shipgen_class_weights.cache_clear()
        monkeypatch.setattr(shipgen_gen, "get_ship_classes", lambda *args, **kwargs: ["patrol"])

        rng = random.Random(123)
        results = {
            _pick_svg(
                None,
                rng,
                length_range=(35.0, 80.0),
                offnadir_deg=7.0,
                sensor_az_ship_deg=123.0,
                shipgen_kwargs={"deck_scatter_density": 2.5},
            )
            for _ in range(12)
        }

        assert len(results) >= 4
        for svg_text in results:
            root = ET.fromstring(svg_text)
            assert root.get("data-ship-class") == "patrol"


class TestShipSizeDistribution:
    """compute_ship_pixel_size の長さ分布が対数一様になっているか。"""

    def test_log_uniform_more_small_ships(self) -> None:
        """10-150m 範囲で生成すると中央値が線形一様より小さくなる。"""
        rng = random.Random(42)
        lengths = []
        for _ in range(2000):
            _bw, lh = compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng, length_range=(10.0, 150.0),
            )
            lengths.append(lh)
        median = sorted(lengths)[len(lengths) // 2]
        assert median < 55, f"Median {median} too high - distribution not log-uniform"

    def test_log_uniform_still_produces_large(self) -> None:
        """大きな船もゼロではない。"""
        rng = random.Random(0)
        lengths = []
        for _ in range(500):
            _bw, lh = compute_ship_pixel_size(
                "carrier", 5.0, 1.0, rng, length_range=(10.0, 300.0),
            )
            lengths.append(lh)
        assert max(lengths) > 250


class TestLengthExponent:
    """length_exponent パラメータによるサイズ分布制御のテスト。"""

    def test_exponent_1_is_log_uniform(self) -> None:
        """exponent=1.0 は従来の対数一様分布と同等。"""
        rng = random.Random(42)
        lengths = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng,
                length_range=(10.0, 150.0), length_exponent=1.0,
            )[1]
            for _ in range(2000)
        ]
        median = sorted(lengths)[len(lengths) // 2]
        assert median < 55, f"Median {median} too high for log-uniform"

    def test_exponent_gt1_more_small(self) -> None:
        """exponent>1 にすると中央値が下がる（小さい船が増える）。"""
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        lengths_1 = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng1,
                length_range=(10.0, 150.0), length_exponent=1.0,
            )[1]
            for _ in range(2000)
        ]
        lengths_3 = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng2,
                length_range=(10.0, 150.0), length_exponent=3.0,
            )[1]
            for _ in range(2000)
        ]
        median_1 = sorted(lengths_1)[len(lengths_1) // 2]
        median_3 = sorted(lengths_3)[len(lengths_3) // 2]
        assert median_3 < median_1

    def test_exponent_lt1_more_large(self) -> None:
        """exponent<1 にすると中央値が上がる（大きい船が増える）。"""
        rng1 = random.Random(7)
        rng2 = random.Random(7)
        lengths_1 = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng1,
                length_range=(10.0, 150.0), length_exponent=1.0,
            )[1]
            for _ in range(2000)
        ]
        lengths_05 = [
            compute_ship_pixel_size(
                "patrol", 5.0, 1.0, rng2,
                length_range=(10.0, 150.0), length_exponent=0.3,
            )[1]
            for _ in range(2000)
        ]
        median_1 = sorted(lengths_1)[len(lengths_1) // 2]
        median_05 = sorted(lengths_05)[len(lengths_05) // 2]
        assert median_05 > median_1


class TestNaturalLbRatio:
    """_natural_lb_ratio の物理的妥当性検証。"""

    def test_small_ship_low_lb(self) -> None:
        """5m ディンギー等、小型船は低い lb_ratio。"""
        assert _natural_lb_ratio(5.0) < 4.0

    def test_large_ship_higher_lb(self) -> None:
        """200m 級の大型船はより高い lb_ratio。"""
        assert _natural_lb_ratio(200.0) > _natural_lb_ratio(20.0)

    def test_capped_at_10(self) -> None:
        """10 が上限。"""
        assert _natural_lb_ratio(10000.0) == 10.0

    def test_monotone_increasing(self) -> None:
        """lb_ratio は船の長さに対して単調増加する。"""
        lengths = [5.0, 20.0, 50.0, 100.0, 200.0, 300.0]
        values = [_natural_lb_ratio(length) for length in lengths]
        for a, b in zip(values, values[1:]):
            assert a <= b


class TestSvgLbWeight:
    """_svg_lb_weight の重み計算の検証。"""

    def test_natural_lb_gets_full_weight(self) -> None:
        """自然な lb_ratio の船は重み 1.0。"""
        lb = _natural_lb_ratio(15.0)
        assert _svg_lb_weight(lb, 15.0) == 1.0

    def test_excess_lb_gets_lower_weight(self) -> None:
        """lb_ratio が自然値の 1.5 倍を超えると重みが下がる。"""
        w_bad = _svg_lb_weight(12.0, 10.0)
        w_good = _svg_lb_weight(3.5, 10.0)
        assert w_bad < w_good

    def test_hard_reject_above_outer_band(self) -> None:
        """natural の _LB_OUTER_BAND_MULTIPLIER 倍を超える lb_ratio は hard-reject (weight=0.0)。"""
        assert _svg_lb_weight(15.0, 5.0) == 0.0

    def test_within_outer_band_has_positive_weight(self) -> None:
        """natural の _LB_OUTER_BAND_MULTIPLIER 倍以内の lb_ratio は正の重みを返す。"""
        assert _svg_lb_weight(6.0, 5.0) > 0.0

    def test_hard_reject_boundary_small_ship(self) -> None:
        """小型船 (15m) で、過度に細長い lb は 0.0 になる。"""
        assert _svg_lb_weight(10.0, 15.0) == 0.0
        assert _svg_lb_weight(6.0, 15.0) > 0.0

    def test_small_target_prefers_low_lb(self) -> None:
        """小型船ターゲットでは、低 lb_ratio の SVG が高く評価される。"""
        w_stubby = _svg_lb_weight(4.0, 10.0)
        w_slender = _svg_lb_weight(9.0, 10.0)
        assert w_stubby > w_slender

    def test_hard_reject_below_outer_band(self) -> None:
        """natural / _LB_OUTER_BAND_MULTIPLIER 未満の lb_ratio は hard-reject (weight=0.0)。"""
        # 100m 船 (natural ≈ 6.0): L/B=2 はビーム50m となり非現実的。
        assert _svg_lb_weight(2.0, 100.0) == 0.0

    def test_soft_reject_in_lower_zone(self) -> None:
        """outer-band 〜 inner-band の帯域はソフトペナルティ (0 < w < 1)。"""
        natural = _natural_lb_ratio(100.0)  # 6.0
        lb_in_soft_zone = (
            natural / _LB_OUTER_BAND_MULTIPLIER
            + natural / _MAX_REASONABLE_LB_RATIO_MULTIPLIER
        ) / 2.0
        w = _svg_lb_weight(lb_in_soft_zone, 100.0)
        assert 0.0 < w < 1.0

    def test_lower_penalty_symmetric_with_upper(self) -> None:
        """上下限ペナルティが natural に対して対称になっている。"""
        natural = _natural_lb_ratio(80.0)
        # inner-band の境界から等距離だけ外側に出た点は同じペナルティを受ける
        excess = 0.2
        w_above = _svg_lb_weight(
            natural * _MAX_REASONABLE_LB_RATIO_MULTIPLIER + excess, 80.0
        )
        w_below = _svg_lb_weight(
            natural / _MAX_REASONABLE_LB_RATIO_MULTIPLIER - excess, 80.0
        )
        assert abs(w_above - w_below) < 1e-9

    def test_slender_50m_ship_full_weight(self) -> None:
        """50m×7m の細長い船 (L/B≈7.14) は 50m ターゲットで重み 1.0 を得る。"""
        # 現実に存在する細長い船が inner band に入ることを確認する。
        assert _svg_lb_weight(7.14, 50.0) == pytest.approx(1.0)

    def test_patrol_midpoint_lb_full_weight_at_50m(self) -> None:
        """patrol クラス L/B 中央値 (7.75) が 50m ターゲットで重み 1.0 を得る。"""
        # patrol (L/B 5.5-10.0) の on-demand 生成で 50m 船を downweight しないことを確認。
        assert _svg_lb_weight(7.75, 50.0) == pytest.approx(1.0)

    def test_min_reasonable_lb_ratio_less_than_natural(self) -> None:
        """_min_reasonable_lb_ratio は常に natural より小さい。"""
        for length in [10.0, 50.0, 100.0, 200.0]:
            assert _min_reasonable_lb_ratio(length) < _natural_lb_ratio(length)


class TestLoadSvgMetas:
    """_load_svg_metas のメタデータ読み込みまとめの検証。"""

    def test_reads_lb_ratio(self, tmp_path: pathlib.Path) -> None:
        """SVG ファイルから lb_ratio を正しく読み取る。"""
        svg_content = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 4.5"'
            ' data-ship-class="fishing_trawler" data-lb-ratio="4.5">'
            '<polygon points="0.5,0 0,4.5 1,4.5" fill="#666"/></svg>'
        )
        svg_path = tmp_path / "test_ship.svg"
        svg_path.write_text(svg_content, encoding="utf-8")

        metas = _load_svg_metas([svg_path])
        assert len(metas) == 1
        assert isinstance(metas[0], _SvgMeta)
        assert metas[0].path == svg_path
        assert metas[0].lb_ratio == 4.5

    def test_multiple_files(self, tmp_path: pathlib.Path) -> None:
        """複数ファイルのメタが順番通り返る。"""
        lb_values = [3.8, 6.5, 9.0]
        paths = []
        for i, lb in enumerate(lb_values):
            content = (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 {lb}"'
                f' data-ship-class="patrol" data-lb-ratio="{lb}">'
                f'<polygon points="0.5,0 0,{lb} 1,{lb}" fill="#888"/></svg>'
            )
            path = tmp_path / f"ship_{i}.svg"
            path.write_text(content, encoding="utf-8")
            paths.append(path)

        metas = _load_svg_metas(paths)
        assert [meta.lb_ratio for meta in metas] == lb_values


class TestShipClassId:
    """_ship_class_id による大小クラス判定の検証。"""

    def test_no_threshold_returns_base_id(self) -> None:
        """しきい値なしのとき、常に base class_id を返す。"""
        assert _ship_class_id(100, 10.0, 0, None) == 0

    def test_below_threshold_returns_small(self) -> None:
        """長さがしきい値未満なら small (class_id) を返す。"""
        assert _ship_class_id(5, 10.0, 0, (100.0,)) == 0

    def test_at_threshold_returns_large(self) -> None:
        """長さがしきい値ちょうどなら large (class_id + 1) を返す。"""
        assert _ship_class_id(10, 10.0, 0, (100.0,)) == 1

    def test_above_threshold_returns_large(self) -> None:
        """長さがしきい値超なら large (class_id + 1) を返す。"""
        assert _ship_class_id(15, 10.0, 0, (100.0,)) == 1

    def test_custom_base_class_id(self) -> None:
        """base class_id が 0 以外でも正しく動作する。"""
        assert _ship_class_id(5, 10.0, 2, (100.0,)) == 2
        assert _ship_class_id(15, 10.0, 2, (100.0,)) == 3

    def test_two_thresholds_three_buckets(self) -> None:
        """しきい値2つのとき、長さに応じて 0/1/2 のバケットに割り振られる。"""
        # thresholds: 30m, 80m
        # 5px * 10m = 50m → 30以上80未満 → bucket 1
        assert _ship_class_id(5, 10.0, 0, (30.0, 80.0)) == 1
        # 2px * 10m = 20m → 30未満 → bucket 0
        assert _ship_class_id(2, 10.0, 0, (30.0, 80.0)) == 0
        # 9px * 10m = 90m → 80以上 → bucket 2
        assert _ship_class_id(9, 10.0, 0, (30.0, 80.0)) == 2

    def test_three_thresholds_four_buckets(self) -> None:
        """しきい値3つのとき、4クラスに割り振られる。"""
        # thresholds: 20m, 50m, 100m
        assert _ship_class_id(1, 10.0, 0, (20.0, 50.0, 100.0)) == 0   # 10m < 20
        assert _ship_class_id(3, 10.0, 0, (20.0, 50.0, 100.0)) == 1   # 30m: [20,50)
        assert _ship_class_id(7, 10.0, 0, (20.0, 50.0, 100.0)) == 2   # 70m: [50,100)
        assert _ship_class_id(11, 10.0, 0, (20.0, 50.0, 100.0)) == 3  # 110m ≥ 100

    def test_unsorted_thresholds_behave_same_as_sorted(self) -> None:
        """しきい値の順序に関わらず同じ結果を返す。"""
        assert _ship_class_id(5, 10.0, 0, (80.0, 30.0)) == _ship_class_id(5, 10.0, 0, (30.0, 80.0))
        assert _ship_class_id(9, 10.0, 0, (80.0, 30.0)) == _ship_class_id(9, 10.0, 0, (30.0, 80.0))

    def test_is_cluster_no_threshold_offsets_by_1(self) -> None:
        """しきい値なし・is_cluster=True は solo クラス数 (1) 分オフセットされる。"""
        assert _ship_class_id(100, 10.0, 0, None, is_cluster=True) == 1

    def test_is_cluster_with_one_threshold(self) -> None:
        """しきい値1つ・is_cluster=True は solo クラス数 (2) 分オフセットされる。"""
        # small solo=0 → small_c=2
        assert _ship_class_id(5, 10.0, 0, (100.0,), is_cluster=True) == 2
        # large solo=1 → large_c=3
        assert _ship_class_id(10, 10.0, 0, (100.0,), is_cluster=True) == 3

    def test_is_cluster_false_is_default(self) -> None:
        """is_cluster=False はデフォルトと同一結果を返す。"""
        assert _ship_class_id(5, 10.0, 0, (100.0,), is_cluster=False) == _ship_class_id(5, 10.0, 0, (100.0,))


class TestSizeClassNames:
    """_size_class_names の命名ルール検証。"""

    def test_no_threshold_gives_ship_and_cluster(self) -> None:
        """しきい値なしは ship / ship_c の2クラスを返す。"""
        assert _size_class_names(()) == ["ship", "ship_c"]

    def test_one_threshold_gives_small_large_and_clusters(self) -> None:
        """しきい値1つは solo 2クラス + cluster 2クラスを返す。"""
        assert _size_class_names((50.0,)) == [
            "ship_small", "ship_large",
            "ship_small_c", "ship_large_c",
        ]

    def test_two_thresholds_gives_medium_and_clusters(self) -> None:
        """しきい値2つは solo 3クラス + cluster 3クラスを返す。"""
        assert _size_class_names((30.0, 80.0)) == [
            "ship_small", "ship_medium", "ship_large",
            "ship_small_c", "ship_medium_c", "ship_large_c",
        ]

    def test_three_thresholds_uses_numeric_names(self) -> None:
        """しきい値3つは境界値数値を使った中間クラス名（solo+cluster 合計8クラス）。"""
        names = _size_class_names((20.0, 50.0, 100.0))
        assert names == [
            "ship_small", "ship_20_50", "ship_50_100", "ship_large",
            "ship_small_c", "ship_20_50_c", "ship_50_100_c", "ship_large_c",
        ]

    def test_unsorted_thresholds_produce_sorted_names(self) -> None:
        """しきい値は内部でソートされる。"""
        names = _size_class_names((80.0, 20.0, 50.0))
        assert names == [
            "ship_small", "ship_20_50", "ship_50_80", "ship_large",
            "ship_small_c", "ship_20_50_c", "ship_50_80_c", "ship_large_c",
        ]

    def test_cluster_names_are_solo_names_with_c_suffix(self) -> None:
        """cluster クラス名は solo 名に _c サフィックスを付けたものと一致する。"""
        names = _size_class_names((30.0, 80.0))
        solo = names[:3]
        cluster = names[3:]
        assert cluster == [f"{n}_c" for n in solo]