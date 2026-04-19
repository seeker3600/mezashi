"""Tests for medetect.yolo.expand_obb."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from medetect.yolo.expand_obb import (
    _expand_obb,
    _obb_dimensions,
    _process_label_file,
    expand_obb_dataset,
)


def _label_corners(label_line: str) -> np.ndarray:
    tokens = label_line.strip().split()
    return np.array([float(token) for token in tokens[1:]], dtype=float).reshape(4, 2)


def _overlap_area(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    return float(Polygon(corners_a).intersection(Polygon(corners_b)).area)


class TestObbDimensions:
    def test_axis_aligned_returns_shorter_and_longer(self) -> None:
        """軸に平行なOBBでは短辺がwidth、長辺がheightになる。"""
        corners = np.array(
            [[40, 30], [60, 30], [60, 70], [40, 70]], dtype=float
        )
        width, height = _obb_dimensions(corners)
        assert width == pytest.approx(20.0)
        assert height == pytest.approx(40.0)

    def test_square_returns_equal_dims(self) -> None:
        """正方形のOBBでは幅と高さが等しい。"""
        corners = np.array(
            [[40, 40], [60, 40], [60, 60], [40, 60]], dtype=float
        )
        width, height = _obb_dimensions(corners)
        assert width == pytest.approx(20.0)
        assert height == pytest.approx(20.0)

    def test_rotated_returns_correct_dims(self) -> None:
        """45度回転したOBBでも正しい寸法を返す。"""
        s = np.sqrt(2) / 2
        u = np.array([s * 10, s * 10])
        v = np.array([-s * 20, s * 20])
        center = np.array([50.0, 50.0])
        corners = np.array([
            center - u - v,
            center + u - v,
            center + u + v,
            center - u + v,
        ])
        width, height = _obb_dimensions(corners)
        assert width == pytest.approx(20.0)
        assert height == pytest.approx(40.0)


class TestExpandObb:
    def test_expand_height_axis_aligned(self) -> None:
        """軸に平行なOBBの高さ(長辺)を10px広げる。"""
        corners = np.array(
            [[40, 30], [60, 30], [60, 70], [40, 70]], dtype=float
        )
        result = _expand_obb(corners, expand_width=0, expand_height=10)
        expected = np.array(
            [[40, 25], [60, 25], [60, 75], [40, 75]], dtype=float
        )
        np.testing.assert_allclose(result, expected)

    def test_expand_width_axis_aligned(self) -> None:
        """軸に平行なOBBの幅(短辺)を6px広げる。"""
        corners = np.array(
            [[40, 30], [60, 30], [60, 70], [40, 70]], dtype=float
        )
        result = _expand_obb(corners, expand_width=6, expand_height=0)
        expected = np.array(
            [[37, 30], [63, 30], [63, 70], [37, 70]], dtype=float
        )
        np.testing.assert_allclose(result, expected)

    def test_expand_both_axis_aligned(self) -> None:
        """軸に平行なOBBの幅と高さを同時に広げる。"""
        corners = np.array(
            [[40, 30], [60, 30], [60, 70], [40, 70]], dtype=float
        )
        result = _expand_obb(corners, expand_width=6, expand_height=10)
        expected = np.array(
            [[37, 25], [63, 25], [63, 75], [37, 75]], dtype=float
        )
        np.testing.assert_allclose(result, expected)

    def test_no_expansion_returns_same_corners(self) -> None:
        """expand=0なら同じコーナー座標を返す。"""
        corners = np.array(
            [[40, 30], [60, 30], [60, 70], [40, 70]], dtype=float
        )
        result = _expand_obb(corners, expand_width=0, expand_height=0)
        np.testing.assert_allclose(result, corners)

    def test_expand_rotated_preserves_center(self) -> None:
        """回転したOBBの拡張後もセンターが変わらない。"""
        s = np.sqrt(2) / 2
        u = np.array([s * 10, s * 10])
        v = np.array([-s * 20, s * 20])
        center = np.array([50.0, 50.0])
        corners = np.array([
            center - u - v,
            center + u - v,
            center + u + v,
            center - u + v,
        ])
        result = _expand_obb(corners, expand_width=4, expand_height=6)
        result_center = result.mean(axis=0)
        np.testing.assert_allclose(result_center, center, atol=1e-10)

    def test_expand_rotated_increases_dims(self) -> None:
        """回転したOBBの拡張後に寸法が正しく増加する。"""
        s = np.sqrt(2) / 2
        u = np.array([s * 10, s * 10])
        v = np.array([-s * 20, s * 20])
        center = np.array([50.0, 50.0])
        corners = np.array([
            center - u - v,
            center + u - v,
            center + u + v,
            center - u + v,
        ])
        result = _expand_obb(corners, expand_width=4, expand_height=6)
        width, height = _obb_dimensions(result)
        assert width == pytest.approx(24.0)
        assert height == pytest.approx(46.0)


class TestProcessLabelFile:
    @pytest.fixture()
    def dataset(self, tmp_path):
        """100x100 画像付きのミニデータセットを作る。"""
        from PIL import Image

        ds_root = tmp_path / "dataset"
        img_dir = ds_root / "images" / "train"
        lbl_dir = ds_root / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        Image.new("RGB", (100, 100)).save(img_dir / "test.png")
        return ds_root

    def test_constant_expand_height(self, dataset) -> None:
        """定数でOBBの高さを広げる。"""
        lbl_path = dataset / "labels" / "train" / "test.txt"
        # 20x40 box centered at (50, 50): width=20, height=40
        lbl_path.write_text(
            "0 0.400000 0.300000 0.600000 0.300000 "
            "0.600000 0.700000 0.400000 0.700000\n"
        )

        result = _process_label_file(
            lbl_path,
            img_w=100,
            img_h=100,
            expand_width=0,
            expand_height=10,
            expand_width_weighted=0,
            expand_height_weighted=0,
            median_width=0,
            median_height=0,
        )

        assert result["updated"] == 1
        assert result["labels_expanded"] == 1

        tokens = lbl_path.read_text().strip().split()
        coords = [float(t) for t in tokens[1:]]
        corners = np.array(coords).reshape(4, 2)
        # Height expanded by 10: y goes from 0.3→0.25 and 0.7→0.75
        np.testing.assert_allclose(
            corners,
            [[0.40, 0.25], [0.60, 0.25], [0.60, 0.75], [0.40, 0.75]],
            atol=1e-5,
        )

    def test_weighted_expand_width(self, dataset) -> None:
        """加重付きでOBBの幅を広げる。幅が狭いほど多く広がる。"""
        lbl_path = dataset / "labels" / "train" / "test.txt"
        # Box with width=10 (short), height=40 (long)
        lbl_path.write_text(
            "0 0.450000 0.300000 0.550000 0.300000 "
            "0.550000 0.700000 0.450000 0.700000\n"
        )

        # median_width=20, base=4 → expansion = 4 * (20/10) = 8
        result = _process_label_file(
            lbl_path,
            img_w=100,
            img_h=100,
            expand_width=0,
            expand_height=0,
            expand_width_weighted=4,
            expand_height_weighted=0,
            median_width=20,
            median_height=40,
        )

        assert result["updated"] == 1
        tokens = lbl_path.read_text().strip().split()
        coords = [float(t) for t in tokens[1:]]
        corners = np.array(coords).reshape(4, 2)
        # Width: 10+8=18, center_x=50 → x: (50-9)/100=0.41, (50+9)/100=0.59
        np.testing.assert_allclose(
            corners[:, 0], [0.41, 0.59, 0.59, 0.41], atol=1e-5
        )

    def test_no_change_when_no_expansion(self, dataset) -> None:
        """展開量が0ならファイルは変更されない。"""
        lbl_path = dataset / "labels" / "train" / "test.txt"
        original = (
            "0 0.400000 0.300000 0.600000 0.300000 "
            "0.600000 0.700000 0.400000 0.700000\n"
        )
        lbl_path.write_text(original)

        result = _process_label_file(
            lbl_path,
            img_w=100,
            img_h=100,
            expand_width=0,
            expand_height=0,
            expand_width_weighted=0,
            expand_height_weighted=0,
            median_width=0,
            median_height=0,
        )

        assert result["updated"] == 0
        assert lbl_path.read_text() == original

    def test_clamps_to_image_bounds(self, dataset) -> None:
        """画像境界を越えるOBBは[0,1]にクランプされる。"""
        lbl_path = dataset / "labels" / "train" / "test.txt"
        # Box near the edge: center(90,50), width=10, height=40
        lbl_path.write_text(
            "0 0.850000 0.300000 0.950000 0.300000 "
            "0.950000 0.700000 0.850000 0.700000\n"
        )

        _process_label_file(
            lbl_path,
            img_w=100,
            img_h=100,
            expand_width=30,
            expand_height=0,
            expand_width_weighted=0,
            expand_height_weighted=0,
            median_width=0,
            median_height=0,
        )

        tokens = lbl_path.read_text().strip().split()
        coords = [float(t) for t in tokens[1:]]
        # All x-coordinates clamped to [0.0, 1.0]
        assert all(0.0 <= c <= 1.0 for c in coords)

    def test_avoid_overlap_scales_later_box_back(self, dataset) -> None:
        """衝突回避時は後続OBBを縮退させて重なりを防ぐ。"""
        lbl_path = dataset / "labels" / "train" / "test.txt"
        lbl_path.write_text(
            "0 0.200000 0.300000 0.400000 0.300000 "
            "0.400000 0.700000 0.200000 0.700000\n"
            "1 0.460000 0.300000 0.660000 0.300000 "
            "0.660000 0.700000 0.460000 0.700000\n"
        )

        result = _process_label_file(
            lbl_path,
            img_w=100,
            img_h=100,
            expand_width=10,
            expand_height=0,
            expand_width_weighted=0,
            expand_height_weighted=0,
            median_width=0,
            median_height=0,
            avoid_overlap=True,
        )

        assert result["updated"] == 1
        assert result["labels_expanded"] >= 1

        lines = lbl_path.read_text().strip().splitlines()
        first = _label_corners(lines[0])
        second = _label_corners(lines[1])

        first_width = (first[:, 0].max() - first[:, 0].min()) * 100
        second_width = (second[:, 0].max() - second[:, 0].min()) * 100

        assert first_width == pytest.approx(30.0, abs=1e-4)
        assert 20.0 < second_width < 30.0
        assert _overlap_area(first, second) == pytest.approx(0.0, abs=1e-8)

    def test_avoid_overlap_skips_initially_overlapping_boxes(self, dataset) -> None:
        """初期状態で重なるOBBは衝突回避モードでは据え置く。"""
        lbl_path = dataset / "labels" / "train" / "test.txt"
        original = (
            "0 0.200000 0.300000 0.400000 0.300000 "
            "0.400000 0.700000 0.200000 0.700000\n"
            "1 0.350000 0.300000 0.550000 0.300000 "
            "0.550000 0.700000 0.350000 0.700000\n"
        )
        lbl_path.write_text(original)

        result = _process_label_file(
            lbl_path,
            img_w=100,
            img_h=100,
            expand_width=10,
            expand_height=0,
            expand_width_weighted=0,
            expand_height_weighted=0,
            median_width=0,
            median_height=0,
            avoid_overlap=True,
        )

        assert result["updated"] == 0
        assert result["labels_expanded"] == 0
        assert lbl_path.read_text() == original

    def test_avoid_overlap_handles_rotated_boxes(self, dataset) -> None:
        """回転OBBでも中心を保ったまま重なりを回避する。"""
        lbl_path = dataset / "labels" / "train" / "test.txt"

        def rotated_box(cx: float, cy: float, w: float, h: float, angle_deg: float) -> str:
            angle_rad = np.deg2rad(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            hw = w / 2.0
            hh = h / 2.0
            base = np.array([
                [-hw, -hh],
                [hw, -hh],
                [hw, hh],
                [-hw, hh],
            ])
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = base @ rot.T + np.array([cx, cy])
            flat = " ".join(f"{value / 100.0:.6f}" for value in corners.flatten())
            return flat

        lbl_path.write_text(
            f"0 {rotated_box(35, 50, 20, 40, 30)}\n"
            f"1 {rotated_box(63, 50, 20, 40, 30)}\n"
        )

        result = _process_label_file(
            lbl_path,
            img_w=100,
            img_h=100,
            expand_width=12,
            expand_height=0,
            expand_width_weighted=0,
            expand_height_weighted=0,
            median_width=0,
            median_height=0,
            avoid_overlap=True,
        )

        assert result["updated"] == 1
        assert result["labels_expanded"] >= 1

        lines = lbl_path.read_text().strip().splitlines()
        first = _label_corners(lines[0])
        second = _label_corners(lines[1])

        assert _overlap_area(first, second) == pytest.approx(0.0, abs=1e-8)
        np.testing.assert_allclose(first.mean(axis=0), [0.35, 0.5], atol=1e-5)
        np.testing.assert_allclose(second.mean(axis=0), [0.63, 0.5], atol=1e-5)


class TestExpandObbDataset:
    def test_full_pipeline(self, tmp_path) -> None:
        """データセット全体でOBB拡張パイプラインが動作する。"""
        from PIL import Image

        ds_root = tmp_path / "dataset"
        img_dir = ds_root / "images" / "train"
        lbl_dir = ds_root / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        Image.new("RGB", (100, 100)).save(img_dir / "img1.png")
        Image.new("RGB", (100, 100)).save(img_dir / "img2.png")

        (lbl_dir / "img1.txt").write_text(
            "0 0.400000 0.300000 0.600000 0.300000 "
            "0.600000 0.700000 0.400000 0.700000\n"
        )
        (lbl_dir / "img2.txt").write_text(
            "1 0.450000 0.300000 0.550000 0.300000 "
            "0.550000 0.700000 0.450000 0.700000\n"
        )

        stats = expand_obb_dataset(
            ds_root,
            expand_height=10,
            expand_width=0,
            max_workers=1,
        )

        assert stats["files_processed"] == 2
        assert stats["files_updated"] == 2
        assert stats["labels_expanded"] == 2

    def test_weighted_pipeline(self, tmp_path) -> None:
        """加重付きOBB拡張で中央値ベースの重み付けが効く。"""
        from PIL import Image

        ds_root = tmp_path / "dataset"
        img_dir = ds_root / "images" / "train"
        lbl_dir = ds_root / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        Image.new("RGB", (100, 100)).save(img_dir / "narrow.png")
        Image.new("RGB", (100, 100)).save(img_dir / "wide.png")

        # narrow: width=10, height=40
        (lbl_dir / "narrow.txt").write_text(
            "0 0.450000 0.300000 0.550000 0.300000 "
            "0.550000 0.700000 0.450000 0.700000\n"
        )
        # wide: width=30, height=40
        (lbl_dir / "wide.txt").write_text(
            "0 0.350000 0.300000 0.650000 0.300000 "
            "0.650000 0.700000 0.350000 0.700000\n"
        )

        stats = expand_obb_dataset(
            ds_root,
            expand_width_weighted=4,
            max_workers=1,
        )

        assert stats["files_processed"] == 2
        assert stats["files_updated"] == 2

        # Check that narrow box got more expansion than wide box
        narrow_tokens = (lbl_dir / "narrow.txt").read_text().strip().split()
        wide_tokens = (lbl_dir / "wide.txt").read_text().strip().split()
        narrow_corners = np.array([float(t) for t in narrow_tokens[1:]]).reshape(4, 2)
        wide_corners = np.array([float(t) for t in wide_tokens[1:]]).reshape(4, 2)

        narrow_w = narrow_corners[:, 0].max() - narrow_corners[:, 0].min()
        wide_w = wide_corners[:, 0].max() - wide_corners[:, 0].min()

        # narrow original width: 0.10 → expanded more
        # wide original width: 0.30 → expanded less
        # Both should be larger than original
        assert narrow_w > 0.10
        assert wide_w > 0.30
        # Narrow expansion ratio should be larger
        narrow_expansion = narrow_w - 0.10
        wide_expansion = wide_w - 0.30
        assert narrow_expansion > wide_expansion

    def test_no_labels_returns_empty_stats(self, tmp_path) -> None:
        """ラベルがない場合は空のstatsを返す。"""
        ds_root = tmp_path / "dataset"
        (ds_root / "labels").mkdir(parents=True)

        stats = expand_obb_dataset(ds_root, expand_height=5)
        assert stats["files_processed"] == 0

    def test_missing_labels_dir_raises(self, tmp_path) -> None:
        """labelsディレクトリが無い場合はFileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            expand_obb_dataset(tmp_path, expand_height=5)

    def test_first_run_creates_backup(self, tmp_path) -> None:
        """初回実行でlabels_before_expandにバックアップを作成する。"""
        from PIL import Image

        ds_root = tmp_path / "dataset"
        img_dir = ds_root / "images" / "train"
        lbl_dir = ds_root / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        Image.new("RGB", (100, 100)).save(img_dir / "img1.png")
        original = (
            "0 0.400000 0.300000 0.600000 0.300000 "
            "0.600000 0.700000 0.400000 0.700000\n"
        )
        (lbl_dir / "img1.txt").write_text(original)

        expand_obb_dataset(ds_root, expand_height=10, max_workers=1)

        backup_path = ds_root / "labels_before_expand" / "train" / "img1.txt"
        assert backup_path.exists()
        assert backup_path.read_text() == original

    def test_second_run_restores_from_backup(self, tmp_path) -> None:
        """2回目以降はバックアップから復元してから再拡張する。"""
        from PIL import Image

        ds_root = tmp_path / "dataset"
        img_dir = ds_root / "images" / "train"
        lbl_dir = ds_root / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        Image.new("RGB", (100, 100)).save(img_dir / "img1.png")
        original = (
            "0 0.400000 0.300000 0.600000 0.300000 "
            "0.600000 0.700000 0.400000 0.700000\n"
        )
        (lbl_dir / "img1.txt").write_text(original)

        # First run: expand by 10
        expand_obb_dataset(ds_root, expand_height=10, max_workers=1)
        tokens1 = (lbl_dir / "img1.txt").read_text().strip().split()
        corners1 = np.array([float(t) for t in tokens1[1:]]).reshape(4, 2)

        # Second run: expand by 20 (should re-expand from original, not stack)
        expand_obb_dataset(ds_root, expand_height=20, max_workers=1)
        tokens2 = (lbl_dir / "img1.txt").read_text().strip().split()
        corners2 = np.array([float(t) for t in tokens2[1:]]).reshape(4, 2)

        # Height 10 expansion: y 0.3→0.25, 0.7→0.75
        np.testing.assert_allclose(
            corners1[:, 1], [0.25, 0.25, 0.75, 0.75], atol=1e-5
        )
        # Height 20 expansion from original: y 0.3→0.20, 0.7→0.80
        np.testing.assert_allclose(
            corners2[:, 1], [0.20, 0.20, 0.80, 0.80], atol=1e-5
        )

    def test_backup_not_overwritten_on_second_run(self, tmp_path) -> None:
        """2回目の実行でバックアップは上書きされない（元のデータが保持される）。"""
        from PIL import Image

        ds_root = tmp_path / "dataset"
        img_dir = ds_root / "images" / "train"
        lbl_dir = ds_root / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        Image.new("RGB", (100, 100)).save(img_dir / "img1.png")
        original = (
            "0 0.400000 0.300000 0.600000 0.300000 "
            "0.600000 0.700000 0.400000 0.700000\n"
        )
        (lbl_dir / "img1.txt").write_text(original)

        expand_obb_dataset(ds_root, expand_height=10, max_workers=1)
        expand_obb_dataset(ds_root, expand_height=20, max_workers=1)

        backup_path = ds_root / "labels_before_expand" / "train" / "img1.txt"
        assert backup_path.read_text() == original
