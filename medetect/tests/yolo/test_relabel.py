from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import pytest

from medetect.yolo.__main__ import main
from medetect.yolo.relabel import (
    _normalize_merges,
    _relabel_line,
    relabel_yolo_detect_dataset,
    relabel_yolo_detect_labels,
)


def _install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch, datasets_dir: Path) -> None:
    fake_module = types.ModuleType("ultralytics")
    fake_module.settings = {"datasets_dir": str(datasets_dir)}
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)


def test_relabel_yolo_detect_dataset_resolves_relative_path_with_ultralytics_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets_dir = tmp_path / "datasets"
    dataset_root = datasets_dir / "toy"
    labels_dir = dataset_root / "labels" / "train"
    labels_dir.mkdir(parents=True)
    (labels_dir / "83.txt").write_text(
        "5 0.013770 0.086740 0.004442 0.003315\n"
        "10 0.017767 0.078269 0.004146 0.004788\n"
        "13 0.040569 0.067035 0.003553 0.003683\n"
        "48 0.043974 0.006446 0.037607 0.012891\n",
        encoding="utf-8",
    )
    (labels_dir / "84.txt").write_text(
        "11 0.1 0.2 0.3 0.4\n"
        "12 0.2 0.3 0.4 0.5\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "toy.yaml"
    config_path.write_text(
        "path: toy\n"
        "merges:\n"
        "  10: 1\n"
        "  11: 1\n"
        "  13: -1\n",
        encoding="utf-8",
    )
    _install_fake_ultralytics(monkeypatch, datasets_dir)

    stats = relabel_yolo_detect_dataset(config_path)

    assert stats == {
        "files_processed": 2,
        "files_updated": 2,
        "labels_reassigned": 2,
        "labels_dropped": 1,
        "empty_labels": 0,
        "images_removed": 0,
    }
    assert (labels_dir / "83.txt").read_text(encoding="utf-8") == (
        "5 0.013770 0.086740 0.004442 0.003315\n"
        "1 0.017767 0.078269 0.004146 0.004788\n"
        "48 0.043974 0.006446 0.037607 0.012891\n"
    )
    assert (labels_dir / "84.txt").read_text(encoding="utf-8") == (
        "1 0.1 0.2 0.3 0.4\n"
        "12 0.2 0.3 0.4 0.5\n"
    )


def test_relabel_yolo_detect_dataset_supports_absolute_dataset_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "absolute"
    labels_dir = dataset_root / "labels" / "val"
    labels_dir.mkdir(parents=True)
    (labels_dir / "sample.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n"
        "2 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "absolute.yaml"
    config_path.write_text(
        f"path: {dataset_root.as_posix()}\n"
        "merges:\n"
        "  2: -1\n",
        encoding="utf-8",
    )

    stats = relabel_yolo_detect_dataset(config_path)

    assert stats == {
        "files_processed": 1,
        "files_updated": 1,
        "labels_reassigned": 0,
        "labels_dropped": 1,
        "empty_labels": 0,
        "images_removed": 0,
    }
    assert (labels_dir / "sample.txt").read_text(encoding="utf-8") == "1 0.5 0.5 0.2 0.2\n"


def test_relabel_yolo_detect_labels_accepts_mapping_from_python(tmp_path: Path) -> None:
    dataset_root = tmp_path / "from-python"
    labels_dir = dataset_root / "labels" / "train"
    labels_dir.mkdir(parents=True)
    (labels_dir / "sample.txt").write_text(
        "1 0.1 0.2 0.3 0.4\n"
        "2 0.2 0.3 0.4 0.5\n"
        "3 0.3 0.4 0.5 0.6\n",
        encoding="utf-8",
    )

    stats = relabel_yolo_detect_labels(
        dataset_root=dataset_root,
        merges={"1": 0, 2: 0, 3: -1},
    )

    assert stats == {
        "files_processed": 1,
        "files_updated": 1,
        "labels_reassigned": 2,
        "labels_dropped": 1,
        "empty_labels": 0,
        "images_removed": 0,
    }
    assert (labels_dir / "sample.txt").read_text(encoding="utf-8") == (
        "0 0.1 0.2 0.3 0.4\n"
        "0 0.2 0.3 0.4 0.5\n"
    )


def test_relabel_yolo_detect_labels_removes_empty_images_when_keep_prob_zero(tmp_path: Path) -> None:
    dataset_root = tmp_path / "drop-empty"
    labels_dir = dataset_root / "labels" / "train"
    images_dir = dataset_root / "images" / "train"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    (labels_dir / "empty.txt").write_text("5 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (images_dir / "empty.jpg").write_text("image", encoding="utf-8")

    stats = relabel_yolo_detect_labels(
        dataset_root=dataset_root,
        merges={5: -1},
        empty_image_keep_prob=0.0,
    )

    assert stats == {
        "files_processed": 1,
        "files_updated": 1,
        "labels_reassigned": 0,
        "labels_dropped": 1,
        "empty_labels": 1,
        "images_removed": 1,
    }
    assert not (labels_dir / "empty.txt").exists()
    assert not (images_dir / "empty.jpg").exists()


def test_relabel_yolo_detect_labels_keeps_empty_images_when_keep_prob_one(tmp_path: Path) -> None:
    dataset_root = tmp_path / "keep-empty"
    labels_dir = dataset_root / "labels" / "train"
    images_dir = dataset_root / "images" / "train"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    (labels_dir / "empty.txt").write_text("5 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (images_dir / "empty.png").write_text("image", encoding="utf-8")

    stats = relabel_yolo_detect_labels(
        dataset_root=dataset_root,
        merges={5: -1},
        empty_image_keep_prob=1.0,
    )

    assert stats == {
        "files_processed": 1,
        "files_updated": 1,
        "labels_reassigned": 0,
        "labels_dropped": 1,
        "empty_labels": 1,
        "images_removed": 0,
    }
    assert (labels_dir / "empty.txt").read_text(encoding="utf-8") == ""
    assert (images_dir / "empty.png").exists()


def test_relabel_yolo_detect_dataset_reads_keep_prob_from_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets_dir = tmp_path / "datasets"
    dataset_root = datasets_dir / "toy"
    labels_dir = dataset_root / "labels" / "train"
    images_dir = dataset_root / "images" / "train"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    (labels_dir / "empty.txt").write_text("13 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (images_dir / "empty.jpg").write_text("image", encoding="utf-8")
    config_path = tmp_path / "toy.yaml"
    config_path.write_text(
        "path: toy\n"
        "empty_image_keep_prob: 0.0\n"
        "merges:\n"
        "  13: -1\n",
        encoding="utf-8",
    )
    _install_fake_ultralytics(monkeypatch, datasets_dir)

    stats = relabel_yolo_detect_dataset(config_path)

    assert stats == {
        "files_processed": 1,
        "files_updated": 1,
        "labels_reassigned": 0,
        "labels_dropped": 1,
        "empty_labels": 1,
        "images_removed": 1,
    }
    assert not (labels_dir / "empty.txt").exists()
    assert not (images_dir / "empty.jpg").exists()


def test_relabel_yolo_detect_labels_keeps_target_ratio_of_empty_images(tmp_path: Path) -> None:
    """empty_image_keep_prob=0.5 のとき、ラベル有り画像と同数のラベル無し画像を残す。"""
    dataset_root = tmp_path / "ratio-half"
    labels_dir = dataset_root / "labels" / "train"
    images_dir = dataset_root / "images" / "train"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    # ラベル有り: 2枚
    (labels_dir / "labeled1.txt").write_text("0 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (labels_dir / "labeled2.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    # ラベル無し (merges で削除): 4枚
    for i in range(1, 5):
        (labels_dir / f"empty{i}.txt").write_text(f"5 0.{i} 0.{i} 0.1 0.1\n", encoding="utf-8")
        (images_dir / f"empty{i}.jpg").write_text("image", encoding="utf-8")

    stats = relabel_yolo_detect_labels(
        dataset_root=dataset_root,
        merges={5: -1},
        empty_image_keep_prob=0.5,
    )

    # target=0.5, n_labeled=2 → keep_count=2, remove=2
    assert stats["empty_labels"] == 4
    assert stats["images_removed"] == 2
    remaining = sum(1 for i in range(1, 5) if (images_dir / f"empty{i}.jpg").exists())
    assert remaining == 2


def test_relabel_yolo_detect_labels_no_removal_when_ratio_already_below_target(
    tmp_path: Path,
) -> None:
    """現在の比率が目標以下の場合は削除しない。"""
    dataset_root = tmp_path / "below-target"
    labels_dir = dataset_root / "labels" / "train"
    images_dir = dataset_root / "images" / "train"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    # ラベル有り: 4枚、ラベル無し: 1枚 → 現在の比率 = 1/5 = 0.2
    for i in range(1, 5):
        (labels_dir / f"labeled{i}.txt").write_text(f"0 0.{i} 0.{i} 0.1 0.1\n", encoding="utf-8")
    (labels_dir / "empty.txt").write_text("5 0.1 0.2 0.3 0.4\n", encoding="utf-8")
    (images_dir / "empty.jpg").write_text("image", encoding="utf-8")

    stats = relabel_yolo_detect_labels(
        dataset_root=dataset_root,
        merges={5: -1},
        empty_image_keep_prob=0.3,  # 目標 0.3 > 現在 0.2 → 削除不要
    )

    assert stats["images_removed"] == 0
    assert (images_dir / "empty.jpg").exists()


def test_cli_main_invokes_relabel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets_dir = tmp_path / "datasets"
    dataset_root = datasets_dir / "toy"
    labels_dir = dataset_root / "labels" / "train"
    labels_dir.mkdir(parents=True)
    (labels_dir / "one.txt").write_text("10 0.1 0.2 0.3 0.4\n", encoding="utf-8")

    config_path = tmp_path / "cli.yaml"
    config_path.write_text(
        "path: toy\n"
        "merges:\n"
        "  10: 1\n",
        encoding="utf-8",
    )
    _install_fake_ultralytics(monkeypatch, datasets_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        ["python", "relabel", "--config", str(config_path)],
    )

    main()
    assert (labels_dir / "one.txt").read_text(encoding="utf-8") == "1 0.1 0.2 0.3 0.4\n"


# ---------------------------------------------------------------------------
# _normalize_merges のサイズベースルール対応テスト
# ---------------------------------------------------------------------------

class TestNormalizeMergesWithSizeRules:
    """_normalize_merges のサイズベースルール対応テスト。"""

    def test_int_value_unchanged(self) -> None:
        """整数値はシンプルなクラス再マッピングとして正規化する。"""
        result = _normalize_merges({0: 1, "5": "-1"})
        assert result == {0: 1, 5: -1}

    def test_dict_value_normalized(self) -> None:
        """辞書値はサイズベースルールとして正規化する。"""
        raw = {"23": {"threshold": "50.0", "below": "0", "above": "2"}}
        result = _normalize_merges(raw)
        assert result == {23: {"threshold": 50.0, "below": 0, "above": 2}}

    def test_mixed_values(self) -> None:
        """整数値と辞書値を混在させられる。"""
        raw = {0: 1, 23: {"threshold": 50.0, "below": 0, "above": 2}}
        result = _normalize_merges(raw)
        assert result[0] == 1
        assert result[23] == {"threshold": 50.0, "below": 0, "above": 2}

    def test_missing_dict_key_raises(self) -> None:
        """辞書値に必須キーが欠けている場合は KeyError。"""
        with pytest.raises(KeyError, match="threshold"):
            _normalize_merges({23: {"below": 0, "above": 2}})


# ---------------------------------------------------------------------------
# _relabel_line with size_merges
# ---------------------------------------------------------------------------

class TestRelabelLineWithSizeMerges:
    """_relabel_line のサイズベースリマッピングのテスト。"""

    def test_below_threshold(self) -> None:
        """長辺が閾値未満 → below クラスへリマップ。"""
        # w=0.01*1000*1.0=10m, h=0.005*1000*1.0=5m → longest=10m < 50 → below=1
        line = "0 0.5 0.5 0.01 0.005"
        result, changed, dropped = _relabel_line(
            line,
            {0: {"threshold": 50.0, "below": 1, "above": 2}},
            img_width=1000, img_height=1000, resolution=1.0,
        )
        assert result == "1 0.5 0.5 0.01 0.005"
        assert changed is True
        assert dropped is False

    def test_above_threshold(self) -> None:
        """長辺が閾値以上 → above クラスへリマップ。"""
        # w=0.1*1000*1.0=100m, h=0.05*1000*1.0=50m → longest=100m >= 50 → above=2
        line = "0 0.5 0.5 0.1 0.05"
        result, changed, dropped = _relabel_line(
            line,
            {0: {"threshold": 50.0, "below": 1, "above": 2}},
            img_width=1000, img_height=1000, resolution=1.0,
        )
        assert result == "2 0.5 0.5 0.1 0.05"
        assert changed is True
        assert dropped is False

    def test_at_exact_threshold(self) -> None:
        """長辺がちょうど閾値 → above (以上)。"""
        # w=0.1*1000*1.0=100m → longest=100m >= 100 → above
        line = "0 0.5 0.5 0.1 0.05"
        result, changed, dropped = _relabel_line(
            line,
            {0: {"threshold": 100.0, "below": 1, "above": 2}},
            img_width=1000, img_height=1000, resolution=1.0,
        )
        assert result == "2 0.5 0.5 0.1 0.05"

    def test_size_rule_on_source_class(self) -> None:
        """source クラスに直接サイズルールを設定できる。"""
        # class 5 has size rule: 10m < 50 → below=1
        line = "5 0.5 0.5 0.01 0.005"
        result, changed, dropped = _relabel_line(
            line,
            {5: {"threshold": 50.0, "below": 1, "above": 2}},
            img_width=1000, img_height=1000, resolution=1.0,
        )
        assert result == "1 0.5 0.5 0.01 0.005"
        assert changed is True

    def test_size_rule_drop(self) -> None:
        """サイズルールで -1 を指定するとドロップされる。"""
        # 10m < 50 → below=-1 → drop
        line = "0 0.5 0.5 0.01 0.005"
        result, changed, dropped = _relabel_line(
            line,
            {0: {"threshold": 50.0, "below": -1, "above": 0}},
            img_width=1000, img_height=1000, resolution=1.0,
        )
        assert result is None
        assert dropped is True

    def test_no_matching_rule(self) -> None:
        """マッチするルールがない場合はそのまま。"""
        line = "0 0.5 0.5 0.1 0.05"
        result, changed, dropped = _relabel_line(
            line,
            {9: {"threshold": 50.0, "below": 1, "above": 2}},
            img_width=1000, img_height=1000, resolution=1.0,
        )
        assert result == "0 0.5 0.5 0.1 0.05"
        assert changed is False

    def test_no_geo_info_skips_size_rule(self) -> None:
        """画像情報がない場合はサイズルールをスキップする。"""
        line = "0 0.5 0.5 0.01 0.005"
        result, changed, dropped = _relabel_line(
            line,
            {0: {"threshold": 50.0, "below": 1, "above": 2}},
            img_width=0, img_height=0, resolution=0.0,
        )
        assert result == "0 0.5 0.5 0.01 0.005"
        assert changed is False


# ---------------------------------------------------------------------------
# relabel_yolo_detect_labels with size rules (integration with GeoTIFF)
# ---------------------------------------------------------------------------

def _create_test_geotiff(
    path: Path, width: int, height: int, resolution: float,
) -> None:
    """テスト用の GeoTIFF を作成する (投影座標系・メートル単位)。"""
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    transform = Affine(resolution, 0, 0, 0, -resolution, 0)
    data = np.zeros((1, height, width), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=1,
        dtype=np.uint8, crs="EPSG:32610",
        transform=transform,
    ) as ds:
        ds.write(data)


class TestRelabelWithSizeMerges:
    """size ルール付き relabel_yolo_detect_labels の統合テスト。"""

    def test_size_based_split(self, tmp_path: Path) -> None:
        """サイズに基づいてクラスを分割できる。"""
        dataset_root = tmp_path / "ds"
        labels_dir = dataset_root / "labels" / "train"
        images_dir = dataset_root / "images" / "train"
        labels_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)

        # Resolution=1.0 m/px, image=1000x1000
        _create_test_geotiff(images_dir / "img.tif", 1000, 1000, 1.0)

        # bbox1: w=0.02, h=0.01 → 20m x 10m → longest=20m (< 50 → below=1)
        # bbox2: w=0.1, h=0.08 → 100m x 80m → longest=100m (>= 50 → above=2)
        (labels_dir / "img.txt").write_text(
            "0 0.5 0.5 0.02 0.01\n"
            "0 0.3 0.3 0.1 0.08\n",
            encoding="utf-8",
        )

        stats = relabel_yolo_detect_labels(
            dataset_root=dataset_root,
            merges={0: {"threshold": 50.0, "below": 1, "above": 2}},
        )

        assert stats["labels_reassigned"] == 2
        content = (labels_dir / "img.txt").read_text(encoding="utf-8")
        assert content == (
            "1 0.5 0.5 0.02 0.01\n"
            "2 0.3 0.3 0.1 0.08\n"
        )

    def test_mixed_int_and_size_rule(self, tmp_path: Path) -> None:
        """整数値の merge とサイズルールを混在できる。"""
        dataset_root = tmp_path / "ds"
        labels_dir = dataset_root / "labels" / "train"
        images_dir = dataset_root / "images" / "train"
        labels_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)

        # Resolution=0.5 m/px, image=2000x2000
        _create_test_geotiff(images_dir / "img.tif", 2000, 2000, 0.5)

        # class 1 → simple remap to 0
        # class 5 → size rule: w=0.01*2000*0.5=10m < 30 → below=1
        (labels_dir / "img.txt").write_text(
            "1 0.5 0.5 0.1 0.1\n"
            "5 0.5 0.5 0.01 0.005\n",
            encoding="utf-8",
        )

        stats = relabel_yolo_detect_labels(
            dataset_root=dataset_root,
            merges={1: 0, 5: {"threshold": 30.0, "below": 1, "above": 2}},
        )

        assert stats["labels_reassigned"] == 2
        content = (labels_dir / "img.txt").read_text(encoding="utf-8")
        assert content == (
            "0 0.5 0.5 0.1 0.1\n"
            "1 0.5 0.5 0.01 0.005\n"
        )

    def test_no_geotiff_skips_size_rule(self, tmp_path: Path) -> None:
        """GeoTIFF がない場合はサイズルールをスキップし通常の merges のみ適用。"""
        dataset_root = tmp_path / "ds"
        labels_dir = dataset_root / "labels" / "train"
        images_dir = dataset_root / "images" / "train"
        labels_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)

        # PNG image (not GeoTIFF) — size rule cannot be applied
        (images_dir / "img.png").write_text("fake", encoding="utf-8")

        (labels_dir / "img.txt").write_text(
            "0 0.5 0.5 0.02 0.01\n",
            encoding="utf-8",
        )

        stats = relabel_yolo_detect_labels(
            dataset_root=dataset_root,
            merges={0: {"threshold": 50.0, "below": 1, "above": 2}},
        )

        # GeoTIFF なし → サイズルールスキップ → 変更なし
        assert stats["labels_reassigned"] == 0
        content = (labels_dir / "img.txt").read_text(encoding="utf-8")
        assert content == "0 0.5 0.5 0.02 0.01\n"


class TestRelabelParallel:
    """並列処理で結果が直列と一致することを確認するテスト。"""

    def test_parallel_produces_same_results(self, tmp_path: Path) -> None:
        """max_workers=2 でも直列と同じ結果を返す。"""
        dataset_root = tmp_path / "parallel"
        labels_dir = dataset_root / "labels" / "train"
        labels_dir.mkdir(parents=True)

        for i in range(10):
            (labels_dir / f"file{i:02d}.txt").write_text(
                f"5 0.{i} 0.{i} 0.1 0.1\n"
                f"10 0.{i} 0.{i} 0.2 0.2\n",
                encoding="utf-8",
            )

        stats = relabel_yolo_detect_labels(
            dataset_root=dataset_root,
            merges={5: 0, 10: -1},
            max_workers=2,
        )

        assert stats["files_processed"] == 10
        assert stats["files_updated"] == 10
        assert stats["labels_reassigned"] == 10
        assert stats["labels_dropped"] == 10

        for i in range(10):
            content = (labels_dir / f"file{i:02d}.txt").read_text(encoding="utf-8")
            assert content == f"0 0.{i} 0.{i} 0.1 0.1\n"

    def test_serial_with_max_workers_one(self, tmp_path: Path) -> None:
        """max_workers=1 で直列実行しても正常に動作する。"""
        dataset_root = tmp_path / "serial"
        labels_dir = dataset_root / "labels" / "train"
        labels_dir.mkdir(parents=True)

        (labels_dir / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (labels_dir / "b.txt").write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        stats = relabel_yolo_detect_labels(
            dataset_root=dataset_root,
            merges={1: 0, 2: 0},
            max_workers=1,
        )

        assert stats["files_processed"] == 2
        assert stats["labels_reassigned"] == 2