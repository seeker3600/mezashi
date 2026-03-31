from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# _build_hyp
# ---------------------------------------------------------------------------


class TestBuildHyp:
    def test_augmentation_params_from_train_kwargs(self) -> None:
        """train_kwargsの拡張パラメータがhypに正しく反映される。"""
        from medetect.yolo.valsplit import _build_hyp

        hyp = _build_hyp()

        assert hyp.degrees == 10.0
        assert hyp.scale == 0.1
        assert hyp.fliplr == 0.5
        assert hyp.flipud == 0.5
        assert hyp.hsv_h == 0.1
        assert hyp.hsv_s == 0.25
        assert hyp.hsv_v == 0.20
        assert hyp.mosaic == 0.0
        assert hyp.erasing == 0.0

    def test_includes_custom_albumentations(self) -> None:
        """カスタムalbumentations変換がaugmentationsに含まれる。"""
        from medetect.yolo.train import custom_transforms
        from medetect.yolo.valsplit import _build_hyp

        hyp = _build_hyp()
        assert hyp.augmentations is custom_transforms

    def test_disable_augs_zeroes_specified_keys(self) -> None:
        """disable_augsで指定したキーが無効値に上書きされる。"""
        from medetect.yolo.valsplit import _build_hyp

        hyp = _build_hyp(disable_augs=["degrees", "scale", "translate", "shear", "perspective"])

        assert hyp.degrees == 0.0
        assert hyp.scale == 0.0
        assert hyp.translate == 0.0
        assert hyp.shear == 0.0
        assert hyp.perspective == 0.0
        # Non-disabled keys keep their train_kwargs values
        assert hyp.fliplr == 0.5
        assert hyp.hsv_h == 0.1

    def test_disable_augs_overrides_train_kwargs(self) -> None:
        """train_kwargsに値があってもdisable_augsで無効化される。"""
        from medetect.yolo.valsplit import _build_hyp

        hyp = _build_hyp(disable_augs=["fliplr", "flipud"])
        assert hyp.fliplr == 0.0
        assert hyp.flipud == 0.0

    def test_disable_augs_unknown_key_raises(self) -> None:
        """存在しない拡張キーを指定するとValueErrorが送出される。"""
        from medetect.yolo.valsplit import _build_hyp

        with pytest.raises(ValueError, match="Unknown augmentation keys"):
            _build_hyp(disable_augs=["nonexistent_key"])

    def test_disable_augs_empty_is_noop(self) -> None:
        """空のdisable_augsはデフォルトと同じ結果になる。"""
        from medetect.yolo.valsplit import _build_hyp

        hyp_default = _build_hyp()
        hyp_empty = _build_hyp(disable_augs=[])
        assert hyp_default.degrees == hyp_empty.degrees
        assert hyp_default.scale == hyp_empty.scale


# ---------------------------------------------------------------------------
# _detect_task
# ---------------------------------------------------------------------------


class TestDetectTask:
    def test_detect_format(self, tmp_path: Path) -> None:
        """4値ラベルをdetectと判定する。"""
        from medetect.yolo.valsplit import _detect_task

        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
        assert _detect_task(tmp_path) == "detect"

    def test_obb_format(self, tmp_path: Path) -> None:
        """8値ラベルをobbと判定する。"""
        from medetect.yolo.valsplit import _detect_task

        (tmp_path / "a.txt").write_text(
            "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
        )
        assert _detect_task(tmp_path) == "obb"

    def test_empty_dir_returns_detect(self, tmp_path: Path) -> None:
        """ラベルが無い場合はdetectをデフォルトとする。"""
        from medetect.yolo.valsplit import _detect_task

        assert _detect_task(tmp_path) == "detect"

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        """空ファイルはスキップして次を読む。"""
        from medetect.yolo.valsplit import _detect_task

        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text(
            "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
        )
        assert _detect_task(tmp_path) == "obb"


# ---------------------------------------------------------------------------
# _save_yolo_labels
# ---------------------------------------------------------------------------


class TestSaveYoloLabels:
    def test_writes_correct_format(self, tmp_path: Path) -> None:
        """YOLO detect形式でラベルが正しく書き出される。"""
        from medetect.yolo.valsplit import _save_yolo_labels

        cls = np.array([[0], [2]])
        bboxes = np.array([[0.5, 0.5, 0.1, 0.2], [0.3, 0.7, 0.05, 0.1]])

        path = tmp_path / "label.txt"
        _save_yolo_labels(path, cls, bboxes)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "0 0.500000 0.500000 0.100000 0.200000"
        assert lines[1] == "2 0.300000 0.700000 0.050000 0.100000"

    def test_writes_obb_format(self, tmp_path: Path) -> None:
        """OBB形式(8座標)でラベルが正しく書き出される。"""
        from medetect.yolo.valsplit import _save_yolo_labels

        cls = np.array([[0]])
        bboxes = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

        path = tmp_path / "label.txt"
        _save_yolo_labels(path, cls, bboxes)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert lines[0] == "0 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000 0.700000 0.800000"

    def test_empty_labels(self, tmp_path: Path) -> None:
        """空のラベルでも空ファイルが作成される。"""
        from medetect.yolo.valsplit import _save_yolo_labels

        cls = np.empty((0, 1))
        bboxes = np.empty((0, 4))

        path = tmp_path / "empty.txt"
        _save_yolo_labels(path, cls, bboxes)

        assert path.read_text() == ""


# ---------------------------------------------------------------------------
# _update_yaml_paths
# ---------------------------------------------------------------------------


class TestUpdateYamlPaths:
    def test_replaces_value_without_comment(self, tmp_path: Path) -> None:
        """コメントなし行の値が書き換えられる。"""
        from medetect.yolo.valsplit import _update_yaml_paths

        config = tmp_path / "dataset.yaml"
        config.write_text("path: /data\ntrain: images/autosplit_train.txt\nval: images/autosplit_val.txt\n")
        _update_yaml_paths(config, {"train": "images/train", "val": "images/val"})

        text = config.read_text()
        assert "train: images/train" in text
        assert "val: images/val" in text

    def test_preserves_inline_comments(self, tmp_path: Path) -> None:
        """インラインコメントが保持される。"""
        from medetect.yolo.valsplit import _update_yaml_paths

        config = tmp_path / "dataset.yaml"
        config.write_text(
            "path: /data  # dataset root\n"
            "train: images/autosplit_train.txt  # training set\n"
            "val: images/autosplit_val.txt  # validation set\n"
            "nc: 1\n"
        )
        _update_yaml_paths(config, {"train": "images/train", "val": "images/val"})

        text = config.read_text()
        assert "train: images/train  # training set" in text
        assert "val: images/val  # validation set" in text
        assert "path: /data  # dataset root" in text

    def test_preserves_other_keys(self, tmp_path: Path) -> None:
        """対象外のキーは変更されない。"""
        from medetect.yolo.valsplit import _update_yaml_paths

        config = tmp_path / "dataset.yaml"
        original = "path: /data\ntrain: old_train\nval: old_val\nnc: 42\n"
        config.write_text(original)
        _update_yaml_paths(config, {"train": "images/train"})

        text = config.read_text()
        assert "nc: 42" in text
        assert "path: /data" in text
        assert "val: old_val" in text


# ---------------------------------------------------------------------------
# split_train_to_val
# ---------------------------------------------------------------------------


class TestSplitTrainToVal:
    @staticmethod
    def _make_dataset(tmp_path: Path, n_images: int = 10) -> tuple[Path, Path, Path]:
        """Minimal YOLO dataset layout and config for testing."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        config = tmp_path / "dataset.yaml"
        config.write_text(
            f"path: {tmp_path}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n  0: ship\n"
            "nc: 1\n"
        )

        for i in range(n_images):
            (images_dir / f"img_{i:04d}.png").write_bytes(b"fake")
            (labels_dir / f"img_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1")

        return config, images_dir, labels_dir

    @staticmethod
    def _mock_dataset(images_dir: Path, n: int) -> MagicMock:
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=n)
        mock.labels = [
            {"im_file": str(images_dir / f"img_{i:04d}.png")}
            for i in range(n)
        ]

        def _getitem(self_mock: MagicMock, idx: int) -> dict:
            return {
                "img": torch.zeros(3, 640, 640, dtype=torch.uint8),
                "cls": torch.tensor([[0]], dtype=torch.float32),
                "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32),
            }

        mock.__getitem__ = _getitem
        return mock

    def test_creates_augmented_val_files(self, tmp_path: Path) -> None:
        """train画像が拡張されてvalディレクトリに保存される。"""
        config, images_dir, labels_dir = self._make_dataset(tmp_path, 10)
        mock = self._mock_dataset(images_dir, 10)

        with (
            patch("medetect.yolo.valsplit.YOLODataset", return_value=mock),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=42)

        val_images = sorted((tmp_path / "images" / "val").iterdir())
        val_labels = sorted((tmp_path / "labels" / "val").iterdir())

        assert len(val_images) == 2
        assert len(val_labels) == 2

        # Originals moved to val_before (not remaining in train)
        remaining_imgs = sorted(images_dir.iterdir())
        remaining_lbls = sorted(labels_dir.iterdir())
        assert len(remaining_imgs) == 8
        assert len(remaining_lbls) == 8

        # val_before preserves the originals
        assert len(sorted((tmp_path / "images" / "val_before").iterdir())) == 2
        assert len(sorted((tmp_path / "labels" / "val_before").iterdir())) == 2

    def test_label_content_matches_augmented_output(self, tmp_path: Path) -> None:
        """保存されたラベルが拡張後のbboxと一致する(detect形式)。"""
        config, images_dir, labels_dir = self._make_dataset(tmp_path, 5)
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=5)
        mock.labels = [
            {"im_file": str(images_dir / f"img_{i:04d}.png")}
            for i in range(5)
        ]

        def _getitem(self_mock: MagicMock, idx: int) -> dict:
            return {
                "img": torch.full((3, 640, 640), 128, dtype=torch.uint8),
                "cls": torch.tensor([[0], [1]], dtype=torch.float32),
                "bboxes": torch.tensor(
                    [[0.25, 0.35, 0.12, 0.08], [0.60, 0.70, 0.05, 0.03]],
                    dtype=torch.float32,
                ),
            }

        mock.__getitem__ = _getitem

        with (
            patch("medetect.yolo.valsplit.YOLODataset", return_value=mock),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=0)

        val_labels_dir = tmp_path / "labels" / "val"
        label_files = sorted(val_labels_dir.iterdir())
        assert len(label_files) == 1

        lines = label_files[0].read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "0 0.250000 0.350000 0.120000 0.080000"
        assert lines[1] == "1 0.600000 0.700000 0.050000 0.030000"

    def test_obb_labels_preserved(self, tmp_path: Path) -> None:
        """OBB形式のラベルがxywhrからxyxyxyxyに正しく変換されて保存される。"""
        import math

        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        config = tmp_path / "dataset.yaml"
        config.write_text(
            f"path: {tmp_path}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n  0: ship\n"
            "nc: 1\n"
        )

        n_images = 5
        for i in range(n_images):
            (images_dir / f"img_{i:04d}.png").write_bytes(b"fake")
            # OBB format: class + 8 coords
            (labels_dir / f"img_{i:04d}.txt").write_text(
                "0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4"
            )

        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=n_images)
        mock.labels = [
            {"im_file": str(images_dir / f"img_{i:04d}.png")}
            for i in range(n_images)
        ]

        # Simulate OBB dataset output: bboxes as (N, 5) xywhr
        def _getitem(self_mock: MagicMock, idx: int) -> dict:
            return {
                "img": torch.full((3, 640, 640), 128, dtype=torch.uint8),
                "cls": torch.tensor([[0]], dtype=torch.float32),
                # cx=0.5, cy=0.5, w=0.2, h=0.1, theta=0.0
                "bboxes": torch.tensor(
                    [[0.5, 0.5, 0.2, 0.1, 0.0]],
                    dtype=torch.float32,
                ),
            }

        mock.__getitem__ = _getitem

        with (
            patch("medetect.yolo.valsplit.YOLODataset", return_value=mock),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=0)

        val_labels_dir = tmp_path / "labels" / "val"
        label_files = sorted(val_labels_dir.iterdir())
        assert len(label_files) == 1

        lines = label_files[0].read_text().strip().split("\n")
        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "0"
        # OBB format: 8 coordinate values
        assert len(parts) == 9, f"Expected 9 values (cls + 8 coords), got {len(parts)}"

    def test_updates_dataset_yaml(self, tmp_path: Path) -> None:
        """実行後にdataset.yamlのtrain/valがディレクトリ指定に書き換えられる。"""
        config, images_dir, labels_dir = self._make_dataset(tmp_path, 5)
        # autosplit_*.txtを使ったYAMLに書き換えてから実行
        config.write_text(
            f"path: {tmp_path}\n"
            "train: images/autosplit_train.txt\n"
            "val: images/autosplit_val.txt\n"
            "names:\n  0: ship\n"
        )
        mock = self._mock_dataset(images_dir, 5)

        with (
            patch("medetect.yolo.valsplit.YOLODataset", return_value=mock),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=0)

        import yaml as _yaml

        updated = _yaml.safe_load(config.read_text())
        assert updated["train"] == "images/train"
        assert updated["val"] == "images/val"

    def test_removes_autosplit_txt(self, tmp_path: Path) -> None:
        """実行後にautosplit_*.txtが削除される。"""
        config, images_dir, labels_dir = self._make_dataset(tmp_path, 5)
        images_root = tmp_path / "images"
        train_txt = images_root / "autosplit_train.txt"
        val_txt = images_root / "autosplit_val.txt"
        train_txt.write_text("images/train/img_0000.png\n")
        val_txt.write_text("")

        mock = self._mock_dataset(images_dir, 5)

        with (
            patch("medetect.yolo.valsplit.YOLODataset", return_value=mock),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=0)

        assert not train_txt.exists()
        assert not val_txt.exists()

    def test_no_autosplit_txt_is_ok(self, tmp_path: Path) -> None:
        """autosplit_*.txtが存在しなくてもエラーにならない。"""
        config, images_dir, _ = self._make_dataset(tmp_path, 5)
        mock = self._mock_dataset(images_dir, 5)

        with (
            patch("medetect.yolo.valsplit.YOLODataset", return_value=mock),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=0)  # should not raise

    def test_train_txt_missing_falls_back_to_directory(self, tmp_path: Path) -> None:
        """trainがtxtファイル指定でファイルが存在しない場合は親ディレクトリで代替する。"""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        for i in range(5):
            (images_dir / f"img_{i:04d}.png").write_bytes(b"fake")
            (labels_dir / f"img_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1")

        config = tmp_path / "dataset.yaml"
        config.write_text(
            f"path: {tmp_path}\n"
            "train: images/autosplit_train.txt\n"
            "val: images/autosplit_val.txt\n"
            "names:\n  0: ship\n"
        )

        # autosplit_*.txt は作成しない

        captured: list[str] = []

        def _fake_yolo_dataset(img_path, **kwargs):
            captured.append(img_path)
            mock = self._mock_dataset(images_dir, 5)
            return mock

        with (
            patch("medetect.yolo.valsplit.YOLODataset", side_effect=_fake_yolo_dataset),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=0)

        # txt ではなくその親ディレクトリ（images/）が渡されていること
        assert captured[0] == str(tmp_path / "images")

    def test_invalid_fraction_raises(self, tmp_path: Path) -> None:
        """不正な割合でValueErrorが送出される。"""
        config, _, _ = self._make_dataset(tmp_path)
        from medetect.yolo.valsplit import split_train_to_val

        with pytest.raises(ValueError):
            split_train_to_val(config, fraction=0.0)
        with pytest.raises(ValueError):
            split_train_to_val(config, fraction=1.0)
        with pytest.raises(ValueError):
            split_train_to_val(config, fraction=-0.1)

    def test_seed_reproducibility(self, tmp_path: Path) -> None:
        """同一seedで同じ画像が選択される。"""
        config, images_dir, _ = self._make_dataset(tmp_path, 20)
        mock = self._mock_dataset(images_dir, 20)

        selected: list[list[str]] = []
        for _ in range(2):
            # Re-create files for each run
            for i in range(20):
                (images_dir / f"img_{i:04d}.png").write_bytes(b"fake")
                (images_dir.parent.parent / "labels" / "train" / f"img_{i:04d}.txt").write_text(
                    "0 0.5 0.5 0.1 0.1"
                )

            with (
                patch("medetect.yolo.valsplit.YOLODataset", return_value=mock),
                patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
            ):
                from medetect.yolo.valsplit import split_train_to_val

                split_train_to_val(config, fraction=0.1, seed=123)

            moved = sorted(
                f.stem for f in (tmp_path / "images" / "val").iterdir()
            )
            selected.append(moved)

            # Clean val for next run
            for f in (tmp_path / "images" / "val").iterdir():
                f.unlink()
            for f in (tmp_path / "labels" / "val").iterdir():
                f.unlink()

            # val_beforeもクリアして次回も初回実行扱いにする
            val_before_img = tmp_path / "images" / "val_before"
            val_before_lbl = tmp_path / "labels" / "val_before"
            if val_before_img.exists():
                for f in val_before_img.iterdir():
                    f.unlink()
            if val_before_lbl.exists():
                for f in val_before_lbl.iterdir():
                    f.unlink()

        assert selected[0] == selected[1]

    def test_val_before_created_on_first_run(self, tmp_path: Path) -> None:
        """初回実行時にval_beforeスプリットが作成され、元ファイルが保存される。"""
        config, images_dir, labels_dir = self._make_dataset(tmp_path, 10)
        mock = self._mock_dataset(images_dir, 10)

        with (
            patch("medetect.yolo.valsplit.YOLODataset", return_value=mock),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=42)

        val_before_images = tmp_path / "images" / "val_before"
        val_before_labels = tmp_path / "labels" / "val_before"
        assert val_before_images.exists()
        assert val_before_labels.exists()
        assert len(list(val_before_images.iterdir())) == 2
        assert len(list(val_before_labels.iterdir())) == 2

        # val_beforeのファイル名はtrain由来のものと一致する
        before_stems = {f.stem for f in val_before_images.iterdir()}
        val_stems = {f.stem for f in (tmp_path / "images" / "val").iterdir()}
        assert before_stems == val_stems

    def test_subsequent_run_uses_val_before(self, tmp_path: Path) -> None:
        """2回目以降はval_beforeをソースとしてvalを再生成し、trainを変更しない。"""
        config, images_dir, labels_dir = self._make_dataset(tmp_path, 10)

        # val_beforeスプリットが既にある状態を用意(初回済み)
        val_before_images = tmp_path / "images" / "val_before"
        val_before_labels = tmp_path / "labels" / "val_before"
        val_before_images.mkdir(parents=True)
        val_before_labels.mkdir(parents=True)
        for i in range(3):
            (val_before_images / f"vb_{i:04d}.png").write_bytes(b"fake")
            (val_before_labels / f"vb_{i:04d}.txt").write_text("0 0.5 0.5 0.1 0.1")

        # val_beforeを指すmock
        mock = MagicMock()
        mock.__len__ = MagicMock(return_value=3)
        mock.labels = [
            {"im_file": str(val_before_images / f"vb_{i:04d}.png")}
            for i in range(3)
        ]

        def _getitem(self_mock: MagicMock, idx: int) -> dict:
            return {
                "img": torch.zeros(3, 640, 640, dtype=torch.uint8),
                "cls": torch.tensor([[0]], dtype=torch.float32),
                "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], dtype=torch.float32),
            }

        mock.__getitem__ = _getitem

        train_count_before = len(list(images_dir.iterdir()))
        captured_img_path: list[str] = []

        def _fake_yolo_dataset(img_path, **kwargs):
            captured_img_path.append(img_path)
            return mock

        with (
            patch("medetect.yolo.valsplit.YOLODataset", side_effect=_fake_yolo_dataset),
            patch("medetect.yolo.valsplit._build_hyp", return_value=MagicMock()),
        ):
            from medetect.yolo.valsplit import split_train_to_val

            split_train_to_val(config, fraction=0.2, seed=0)  # fraction is ignored

        # val_beforeのパスでYOLODatasetが呼ばれること
        assert captured_img_path[0] == str(val_before_images)

        # val には3ファイル生成される
        val_images = sorted((tmp_path / "images" / "val").iterdir())
        assert len(val_images) == 3

        # train は変更されない
        assert len(list(images_dir.iterdir())) == train_count_before

        # val_before は保持される
        assert len(list(val_before_images.iterdir())) == 3


# ---------------------------------------------------------------------------
# CLI – valsplit subcommand
# ---------------------------------------------------------------------------


class TestCliValsplit:
    def test_cli_calls_split_train_to_val(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLIからsplit_train_to_valが正しい引数で呼び出される。"""
        from medetect.yolo.__main__ import main

        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "valsplit", "--config", "ds.yaml", "--fraction", "0.15"],
        )
        with patch("medetect.yolo.__main__.split_train_to_val") as mock_fn:
            main()
        mock_fn.assert_called_once_with(
            config=Path("ds.yaml"),
            fraction=0.15,
            imgsz=640,
            seed=None,
            disable_augs=[],
        )

    def test_cli_optional_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLIのオプション引数が正しく渡される。"""
        from medetect.yolo.__main__ import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog", "valsplit",
                "--config", "ds.yaml",
                "--fraction", "0.1",
                "--imgsz", "1280",
                "--seed", "42",
            ],
        )
        with patch("medetect.yolo.__main__.split_train_to_val") as mock_fn:
            main()
        mock_fn.assert_called_once_with(
            config=Path("ds.yaml"),
            fraction=0.1,
            imgsz=1280,
            seed=42,
            disable_augs=[],
        )

    def test_cli_disable_augs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--disable-augsで指定した拡張キーがリストとして渡される。"""
        from medetect.yolo.__main__ import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog", "valsplit",
                "--config", "ds.yaml",
                "--fraction", "0.2",
                "--disable-augs", "degrees", "scale", "translate",
            ],
        )
        with patch("medetect.yolo.__main__.split_train_to_val") as mock_fn:
            main()
        mock_fn.assert_called_once_with(
            config=Path("ds.yaml"),
            fraction=0.2,
            imgsz=640,
            seed=None,
            disable_augs=["degrees", "scale", "translate"],
        )
