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


# ---------------------------------------------------------------------------
# _save_yolo_labels
# ---------------------------------------------------------------------------


class TestSaveYoloLabels:
    def test_writes_correct_format(self, tmp_path: Path) -> None:
        """YOLO形式でラベルが正しく書き出される。"""
        from medetect.yolo.valsplit import _save_yolo_labels

        cls = np.array([[0], [2]])
        bboxes = np.array([[0.5, 0.5, 0.1, 0.2], [0.3, 0.7, 0.05, 0.1]])

        path = tmp_path / "label.txt"
        _save_yolo_labels(path, cls, bboxes)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "0 0.500000 0.500000 0.100000 0.200000"
        assert lines[1] == "2 0.300000 0.700000 0.050000 0.100000"

    def test_empty_labels(self, tmp_path: Path) -> None:
        """空のラベルでも空ファイルが作成される。"""
        from medetect.yolo.valsplit import _save_yolo_labels

        cls = np.empty((0, 1))
        bboxes = np.empty((0, 4))

        path = tmp_path / "empty.txt"
        _save_yolo_labels(path, cls, bboxes)

        assert path.read_text() == ""


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

        # Originals removed from train
        remaining_imgs = sorted(images_dir.iterdir())
        remaining_lbls = sorted(labels_dir.iterdir())
        assert len(remaining_imgs) == 8
        assert len(remaining_lbls) == 8

    def test_label_content_matches_augmented_output(self, tmp_path: Path) -> None:
        """保存されたラベルが拡張後のbboxと一致する。"""
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

        assert selected[0] == selected[1]


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
        )
