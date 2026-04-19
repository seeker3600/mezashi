from __future__ import annotations

import pathlib
import random

from medetect.datagen.ship import (
    MIN_SHIP_BEAM_PX,
    _SvgMeta,
    _load_svg_metas,
    _natural_lb_ratio,
    _scale_ship_pixel_size,
    _ship_class_id,
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

    def test_hard_reject_above_twice_natural(self) -> None:
        """natural の 2.0 倍を超える lb_ratio は hard-reject (weight=0.0)。"""
        assert _svg_lb_weight(15.0, 5.0) == 0.0

    def test_within_twice_natural_has_positive_weight(self) -> None:
        """natural の 2.0 倍以内の lb_ratio は正の重みを返す。"""
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
        assert _ship_class_id(5, 10.0, 0, 100.0) == 0

    def test_at_threshold_returns_large(self) -> None:
        """長さがしきい値ちょうどなら large (class_id + 1) を返す。"""
        assert _ship_class_id(10, 10.0, 0, 100.0) == 1

    def test_above_threshold_returns_large(self) -> None:
        """長さがしきい値超なら large (class_id + 1) を返す。"""
        assert _ship_class_id(15, 10.0, 0, 100.0) == 1

    def test_custom_base_class_id(self) -> None:
        """base class_id が 0 以外でも正しく動作する。"""
        assert _ship_class_id(5, 10.0, 2, 100.0) == 2
        assert _ship_class_id(15, 10.0, 2, 100.0) == 3