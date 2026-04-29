"""medetect.viewer.obb のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import fiftyone as fo

from medetect.viewer.obb import (
    _build_class_map,
    _image_to_label_path,
    _parse_detect_label_file,
    _parse_obb_label_file,
    detect_task,
    load_yolo_detect_dataset,
    load_yolo_obb_dataset,
)


# ---------------------------------------------------------------------------
# _build_class_map
# ---------------------------------------------------------------------------


class TestBuildClassMap:
    def test_list_input(self) -> None:
        """list 形式の names を {int: str} に変換できる。"""
        result = _build_class_map(["cat", "dog", "bird"])
        assert result == {0: "cat", 1: "dog", 2: "bird"}

    def test_dict_input(self) -> None:
        """dict 形式の names（str キー）を {int: str} に変換できる。"""
        result = _build_class_map({"0": "cat", "2": "bird"})
        assert result == {0: "cat", 2: "bird"}


# ---------------------------------------------------------------------------
# _image_to_label_path
# ---------------------------------------------------------------------------


class TestImageToLabelPath:
    def test_standard_layout(self) -> None:
        """images/train/foo.jpg → labels/train/foo.txt に変換される。"""
        img = Path("/data/images/train/foo.jpg")
        label = _image_to_label_path(img)
        assert label == Path("/data/labels/train/foo.txt")

    def test_no_images_segment(self) -> None:
        """images セグメントがない場合は同階層に .txt を返す。"""
        img = Path("/data/photos/foo.jpg")
        label = _image_to_label_path(img)
        assert label == Path("/data/photos/foo.txt")

    def test_nested_images(self) -> None:
        """深いディレクトリでも正しく変換される。"""
        img = Path("/root/dataset/images/split/sub/bar.png")
        label = _image_to_label_path(img)
        assert label == Path("/root/dataset/labels/split/sub/bar.txt")


# ---------------------------------------------------------------------------
# _parse_obb_label_file
# ---------------------------------------------------------------------------


class TestParseObbLabelFile:
    def _make_label(self, tmp_path: Path, lines: list[str]) -> Path:
        p = tmp_path / "label.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_empty_file_returns_empty_polylines(self, tmp_path: Path) -> None:
        """空ファイルは空の Polylines を返す。"""
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        result = _parse_obb_label_file(p, {0: "ship"})
        assert isinstance(result, fo.Polylines)
        assert len(result.polylines) == 0

    def test_missing_file_returns_empty_polylines(self, tmp_path: Path) -> None:
        """存在しないファイルは空の Polylines を返す。"""
        result = _parse_obb_label_file(tmp_path / "none.txt", {0: "ship"})
        assert len(result.polylines) == 0

    def test_single_obb_line(self, tmp_path: Path) -> None:
        """1行の OBB ラベルが正しく Polyline に変換される。"""
        line = "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
        p = self._make_label(tmp_path, [line])
        result = _parse_obb_label_file(p, {0: "ship"})
        assert len(result.polylines) == 1
        pl = result.polylines[0]
        assert pl.label == "ship"
        assert pl.closed is True
        # points は [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] の入れ子
        pts = pl.points[0]
        assert len(pts) == 4
        assert pts[0] == pytest.approx([0.1, 0.2])
        assert pts[3] == pytest.approx([0.7, 0.8])

    def test_multiple_lines(self, tmp_path: Path) -> None:
        """複数行が全て Polyline に変換される。"""
        lines = [
            "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
            "1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9",
        ]
        p = self._make_label(tmp_path, lines)
        result = _parse_obb_label_file(p, {0: "cat", 1: "dog"})
        assert len(result.polylines) == 2
        assert result.polylines[1].label == "dog"

    def test_unknown_class_uses_id_string(self, tmp_path: Path) -> None:
        """class_map に存在しないクラス ID は文字列化された ID が使われる。"""
        line = "99 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
        p = self._make_label(tmp_path, [line])
        result = _parse_obb_label_file(p, {0: "ship"})
        assert result.polylines[0].label == "99"

    def test_invalid_column_count_raises(self, tmp_path: Path) -> None:
        """9列でない行は ValueError を送出する。"""
        line = "0 0.1 0.2 0.3 0.4"  # 5 columns only
        p = self._make_label(tmp_path, [line])
        with pytest.raises(ValueError, match="9 columns"):
            _parse_obb_label_file(p, {0: "ship"})

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        """空白行は無視される。"""
        lines = ["", "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8", ""]
        p = self._make_label(tmp_path, lines)
        result = _parse_obb_label_file(p, {0: "ship"})
        assert len(result.polylines) == 1


class TestParseDetectLabelFile:
    def _make_label(self, tmp_path: Path, lines: list[str]) -> Path:
        p = tmp_path / "detect.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_single_detect_line(self, tmp_path: Path) -> None:
        """1行の detect ラベルが正しく Detection に変換される。"""
        p = self._make_label(tmp_path, ["0 0.5 0.5 0.2 0.4"])
        result = _parse_detect_label_file(p, {0: "ship"})

        assert len(result.detections) == 1
        detection = result.detections[0]
        assert detection.label == "ship"
        assert detection.bounding_box == pytest.approx([0.4, 0.3, 0.2, 0.4])

    def test_invalid_column_count_raises(self, tmp_path: Path) -> None:
        """5列でない detect ラベルは ValueError を送出する。"""
        p = self._make_label(tmp_path, ["0 0.5 0.5 0.2"])

        with pytest.raises(ValueError, match="5 columns"):
            _parse_detect_label_file(p, {0: "ship"})


# ---------------------------------------------------------------------------
# detect_task
# ---------------------------------------------------------------------------


def _make_dataset_structure(
    tmp_path: Path,
    *,
    n_cols: int,
    split: str = "val",
) -> Path:
    """テスト用の最小 YOLO ディレクトリ構成と YAML を返す。"""
    root = tmp_path / "dataset"
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # ダミー画像 (1x1 PNG)
    img_path = img_dir / "sample.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    # ラベルファイル
    if n_cols == 9:
        label_line = "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
    else:  # 5-column detect
        label_line = "0 0.5 0.5 0.2 0.2"
    (lbl_dir / "sample.txt").write_text(label_line, encoding="utf-8")

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\n{split}: images/{split}\nnames:\n  0: ship\n",
        encoding="utf-8",
    )
    return yaml_path


class TestDetectTask:
    def test_obb_detected(self, tmp_path: Path) -> None:
        """9列ラベルを OBB として検出できる。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=9)
        assert detect_task(yaml_path) == "obb"

    def test_detect_detected(self, tmp_path: Path) -> None:
        """5列ラベルを detect として検出できる。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=5)
        assert detect_task(yaml_path) == "detect"


# ---------------------------------------------------------------------------
# load_yolo_obb_dataset
# ---------------------------------------------------------------------------


class TestLoadYoloObbDataset:
    def test_loads_samples(self, tmp_path: Path) -> None:
        """データセットが正しい件数のサンプルを持つ。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=9)
        ds = load_yolo_obb_dataset(yaml_path, split="val")
        assert len(ds) == 1

    def test_ground_truth_is_polylines(self, tmp_path: Path) -> None:
        """ground_truth フィールドが fo.Polylines 型である。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=9)
        ds = load_yolo_obb_dataset(yaml_path, split="val")
        sample = ds.first()
        assert isinstance(sample["ground_truth"], fo.Polylines)

    def test_split_tag(self, tmp_path: Path) -> None:
        """サンプルに split フィールドが付与される。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=9)
        ds = load_yolo_obb_dataset(yaml_path, split="val")
        assert ds.first()["split"] == "val"

    def test_polyline_has_correct_label(self, tmp_path: Path) -> None:
        """Polyline のラベルが YAML の names と一致する。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=9)
        ds = load_yolo_obb_dataset(yaml_path, split="val")
        pl = ds.first()["ground_truth"].polylines[0]
        assert pl.label == "ship"

    def test_overwrite_removes_old_dataset(self, tmp_path: Path) -> None:
        """overwrite=True の場合、同名データセットを上書きする。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=9)
        name = "test_overwrite_obb"
        ds1 = load_yolo_obb_dataset(yaml_path, split="val", dataset_name=name, overwrite=True)
        ds2 = load_yolo_obb_dataset(yaml_path, split="val", dataset_name=name, overwrite=True)
        assert len(ds2) == 1


class TestLoadYoloDetectDataset:
    def test_loads_samples(self, tmp_path: Path) -> None:
        """detect データセットが正しい件数のサンプルを持つ。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=5)
        ds = load_yolo_detect_dataset(yaml_path, split="val")
        assert len(ds) == 1

    def test_ground_truth_is_detections(self, tmp_path: Path) -> None:
        """ground_truth フィールドが fo.Detections 型である。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=5)
        ds = load_yolo_detect_dataset(yaml_path, split="val")
        sample = ds.first()
        assert isinstance(sample["ground_truth"], fo.Detections)

    def test_detection_has_correct_bbox_and_label(self, tmp_path: Path) -> None:
        """Detection のラベルと bbox が YAML とラベル内容に一致する。"""
        yaml_path = _make_dataset_structure(tmp_path, n_cols=5)
        ds = load_yolo_detect_dataset(yaml_path, split="val")
        detection = ds.first()["ground_truth"].detections[0]
        assert detection.label == "ship"
        assert detection.bounding_box == pytest.approx([0.4, 0.4, 0.2, 0.2])
