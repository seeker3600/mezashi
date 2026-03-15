from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from medetect.shipgen.gen import (
    SHIP_CLASSES,
    _build_hull_polygon,
    _interpolate_hull,
    generate_ship_image,
    generate_ships,
    get_ship_classes,
)


class TestInterpolateHull:
    def test_returns_correct_length(self) -> None:
        """結果の配列長が指定した点数と一致する。"""
        hw = _interpolate_hull("warship", 0.5, 0.15, 100)
        assert len(hw) == 100

    def test_values_in_range(self) -> None:
        """半幅が [0, 0.5] の範囲に収まる。"""
        hw = _interpolate_hull("warship", 0.5, 0.15, 100)
        assert np.all(hw >= 0.0)
        assert np.all(hw <= 0.5)

    def test_bow_starts_narrow_for_warship(self) -> None:
        """軍艦型の艦首は幅ゼロから始まる。"""
        hw = _interpolate_hull("warship", 0.8, 0.15, 100)
        assert hw[0] == pytest.approx(0.0)

    def test_sharper_bow_is_narrower(self) -> None:
        """bow_sharpness が大きいほど前方が細くなる。"""
        hw_blunt = _interpolate_hull("warship", 0.0, 0.15, 100)
        hw_sharp = _interpolate_hull("warship", 1.0, 0.15, 100)
        assert hw_sharp[10] < hw_blunt[10]

    def test_stern_width_applied(self) -> None:
        """指定した艦尾幅が適用される。"""
        hw = _interpolate_hull("warship", 0.5, 0.30, 100)
        assert hw[-1] == pytest.approx(0.30)

    @pytest.mark.parametrize("profile", ["warship", "carrier", "box", "fishing", "fishing_wide"])
    def test_all_profiles_work(self, profile: str) -> None:
        """全プロファイルがエラーなく補間できる。"""
        hw = _interpolate_hull(profile, 0.5, 0.15, 50)
        assert len(hw) == 50
        assert np.all(hw >= 0.0)


class TestBuildHullPolygon:
    def test_polygon_symmetry_no_noise(self) -> None:
        """ノイズなしで船体ポリゴンが左右対称になる。"""
        hw = _interpolate_hull("warship", 0.5, 0.15, 50)
        rng = random.Random(42)
        poly = _build_hull_polygon(50, 20, hw, rng, noise_scale=0.0)

        n = len(hw)
        cx2 = 20  # 2 * center
        for i in range(n):
            rx, ry = poly[i]
            lx, ly = poly[2 * n - 1 - i]
            assert ry == ly
            assert rx + lx == pytest.approx(cx2, abs=2)

    def test_within_image_bounds(self) -> None:
        """ポリゴンの頂点が画像範囲内に収まる。"""
        hw = _interpolate_hull("warship", 0.5, 0.15, 50)
        rng = random.Random(42)
        poly = _build_hull_polygon(50, 20, hw, rng, noise_scale=0.0)

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        assert min(xs) >= 0
        assert max(xs) <= 20
        assert min(ys) >= 0
        assert max(ys) <= 49


class TestGenerateShipImage:
    def test_returns_rgba(self) -> None:
        """生成画像が RGBA モードである。"""
        img = generate_ship_image("patrol", 64, rng=random.Random(42))
        assert img.mode == "RGBA"

    def test_image_height_equals_length(self) -> None:
        """画像の高さが指定した船長に一致する。"""
        img = generate_ship_image("patrol", 64, rng=random.Random(42))
        assert img.size[1] == 64

    def test_has_transparent_and_opaque_pixels(self) -> None:
        """背景が透明で、船体が不透明になっている。"""
        img = generate_ship_image("corvette", 64, rng=random.Random(42))
        arr = np.array(img)
        assert np.any(arr[:, :, 3] == 0), "No transparent pixels"
        assert np.any(arr[:, :, 3] == 255), "No opaque pixels"

    def test_deterministic_with_seed(self) -> None:
        """同一シードで同じ画像が生成される。"""
        img1 = generate_ship_image("frigate", 64, rng=random.Random(99))
        img2 = generate_ship_image("frigate", 64, rng=random.Random(99))
        np.testing.assert_array_equal(np.array(img1), np.array(img2))

    @pytest.mark.parametrize("ship_class", get_ship_classes())
    def test_all_classes_generate(self, ship_class: str) -> None:
        """全艦種でエラーなく画像を生成できる。"""
        img = generate_ship_image(ship_class, 64, rng=random.Random(42))
        assert img.mode == "RGBA"
        assert img.size[1] == 64


class TestGetShipClasses:
    def test_returns_nonempty_sorted_list(self) -> None:
        """利用可能な艦種リストが空でなくソートされている。"""
        classes = get_ship_classes()
        assert len(classes) > 0
        assert classes == sorted(classes)

    def test_matches_registry(self) -> None:
        """レジストリのキーと一致する。"""
        assert set(get_ship_classes()) == set(SHIP_CLASSES)


class TestGenerateShips:
    def test_creates_correct_number_of_files(self, tmp_path: Path) -> None:
        """指定枚数の画像ファイルが出力される。"""
        generate_ships(
            output_dir=tmp_path,
            count=5,
            image_size=(32, 64),
            types={"patrol": 1.0},
            seed=42,
        )
        files = list(tmp_path.glob("*.png"))
        assert len(files) == 5

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        """存在しない出力ディレクトリが自動で作成される。"""
        out = tmp_path / "sub" / "dir"
        generate_ships(
            output_dir=out,
            count=1,
            image_size=(32, 64),
            types={"patrol": 1.0},
            seed=0,
        )
        assert out.is_dir()
        assert len(list(out.glob("*.png"))) == 1

    def test_unknown_class_raises(self, tmp_path: Path) -> None:
        """不明な艦種を指定すると ValueError が発生する。"""
        with pytest.raises(ValueError, match="Unknown ship class"):
            generate_ships(
                output_dir=tmp_path,
                count=1,
                image_size=(32, 64),
                types={"nonexistent_xyz": 1.0},
            )

    def test_default_types_uses_all_classes(self, tmp_path: Path) -> None:
        """types 未指定で全クラスから均等にサンプリングされる。"""
        generate_ships(
            output_dir=tmp_path,
            count=20,
            image_size=(32, 48),
            seed=42,
        )
        files = list(tmp_path.glob("*.png"))
        assert len(files) == 20
