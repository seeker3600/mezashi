from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock, patch


class TestDatagenCommandLogging:
    """datagen CLI のコマンド履歴出力を検証する。"""

    def test_logs_history_to_output_dir(self, tmp_path: Path, monkeypatch) -> None:
        """datagen 完了後に output_dir 直下へ履歴を追記する。"""
        bg_dir = tmp_path / "bg"
        output_dir = tmp_path / "dataset"
        bg_dir.mkdir()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect-datagen",
                "--bg_dir",
                str(bg_dir),
                "--output_dir",
                str(output_dir),
                "--count",
                "3",
            ],
        )

        with (
            patch("medetect.datagen.__main__.generate_dataset", return_value={"images": 3, "ships": 2, "clusters": 1, "skipped": 0}) as mock_generate,
            patch("medetect.datagen.__main__.append_command_history") as mock_log,
        ):
            importlib.import_module("medetect.datagen.__main__").main()

        mock_generate.assert_called_once()
        mock_log.assert_called_once_with(
            output_dir,
            command="datagen",
            result={"images": 3, "ships": 2, "clusters": 1, "skipped": 0},
        )


class TestXviewCommandLogging:
    """xview CLI のコマンド履歴出力を検証する。"""

    def test_logs_history_to_output_dir(self, tmp_path: Path, monkeypatch) -> None:
        """slice 完了後に output_dir 直下へ履歴を追記する。"""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        fake_slice = types.ModuleType("medetect.xview.slice")
        fake_slice.slice_training_images = lambda *args, **kwargs: {"images_processed": 4, "tiles_created": 20}
        monkeypatch.setitem(sys.modules, "medetect.xview.slice", fake_slice)
        sys.modules.pop("medetect.xview", None)
        sys.modules.pop("medetect.xview.__main__", None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect-xview",
                "slice",
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(output_dir),
                "--resolution",
                "0.3",
                "--image_size",
                "640",
            ],
        )

        module = importlib.import_module("medetect.xview.__main__")

        with patch.object(module, "append_command_history") as mock_log:
            stats = module.main()

        assert stats is None
        mock_log.assert_called_once_with(
            output_dir,
            command="xview slice",
            result={"images_processed": 4, "tiles_created": 20},
        )


class TestYoloDatasetCommandLogging:
    """yolo dataset 系 CLI のコマンド履歴出力を検証する。"""

    @staticmethod
    def _write_dataset_yaml(dataset_root: Path, config_path: Path) -> None:
        (dataset_root / "images" / "train").mkdir(parents=True)
        (dataset_root / "labels" / "train").mkdir(parents=True)
        config_path.write_text(
            f"path: {dataset_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "merges:\n"
            "  1: 0\n",
            encoding="utf-8",
        )

    @staticmethod
    def _import_yolo_main(
        monkeypatch,
        *,
        valsplit: MagicMock | None = None,
        expand_obb: MagicMock | None = None,
        relabel: MagicMock | None = None,
        train: MagicMock | None = None,
        tiff2png: MagicMock | None = None,
    ):
        fake_valsplit = types.ModuleType("medetect.yolo.valsplit")
        fake_valsplit.split_train_to_val = valsplit or MagicMock()
        fake_expand = types.ModuleType("medetect.yolo.expand_obb")
        fake_expand.expand_obb_dataset = expand_obb or MagicMock(return_value={})
        fake_relabel = types.ModuleType("medetect.yolo.relabel")
        fake_relabel.relabel_yolo_detect_dataset = relabel or MagicMock(return_value={})
        fake_relabel.relabel_yolo_detect_labels = MagicMock(return_value={})
        fake_train = types.ModuleType("medetect.yolo.train")
        fake_train.train_yolo_model = train or MagicMock()
        fake_tiff2png = types.ModuleType("medetect.yolo.tiff2png")
        fake_tiff2png.convert_tiffs_to_png = tiff2png or MagicMock()

        monkeypatch.setitem(sys.modules, "medetect.yolo.valsplit", fake_valsplit)
        monkeypatch.setitem(sys.modules, "medetect.yolo.expand_obb", fake_expand)
        monkeypatch.setitem(sys.modules, "medetect.yolo.relabel", fake_relabel)
        monkeypatch.setitem(sys.modules, "medetect.yolo.train", fake_train)
        monkeypatch.setitem(sys.modules, "medetect.yolo.tiff2png", fake_tiff2png)
        sys.modules.pop("medetect.yolo", None)
        sys.modules.pop("medetect.yolo.__main__", None)
        return importlib.import_module("medetect.yolo.__main__")

    def test_valsplit_logs_history(self, tmp_path: Path, monkeypatch) -> None:
        """valsplit 完了後に dataset root 直下へ履歴を追記する。"""
        dataset_root = tmp_path / "dataset"
        config_path = tmp_path / "dataset.yaml"
        self._write_dataset_yaml(dataset_root, config_path)
        mock_command = MagicMock()
        monkeypatch.setattr(
            sys,
            "argv",
            ["medetect-yolo", "valsplit", "--config", str(config_path), "--fraction", "0.2"],
        )
        module = self._import_yolo_main(monkeypatch, valsplit=mock_command)

        with patch.object(module, "append_command_history") as mock_log:
            module.main()

        mock_command.assert_called_once()
        mock_log.assert_called_once_with(
            dataset_root.resolve(),
            command="yolo valsplit",
        )

    def test_expand_obb_logs_history_with_stats(self, tmp_path: Path, monkeypatch) -> None:
        """expand-obb 完了後に統計付きで履歴を追記する。"""
        dataset_root = tmp_path / "dataset"
        config_path = tmp_path / "dataset.yaml"
        self._write_dataset_yaml(dataset_root, config_path)
        mock_command = MagicMock(return_value={"files_processed": 5, "files_updated": 3, "labels_expanded": 7})
        monkeypatch.setattr(
            sys,
            "argv",
            ["medetect-yolo", "expand-obb", "--config", str(config_path)],
        )
        module = self._import_yolo_main(monkeypatch, expand_obb=mock_command)

        with patch.object(module, "append_command_history") as mock_log:
            module.main()

        mock_command.assert_called_once()
        mock_log.assert_called_once_with(
            dataset_root.resolve(),
            command="yolo expand-obb",
            result={"files_processed": 5, "files_updated": 3, "labels_expanded": 7},
        )

    def test_relabel_logs_history_with_stats(self, tmp_path: Path, monkeypatch) -> None:
        """relabel 完了後に統計付きで履歴を追記する。"""
        dataset_root = tmp_path / "dataset"
        config_path = tmp_path / "dataset.yaml"
        self._write_dataset_yaml(dataset_root, config_path)
        mock_command = MagicMock(
            return_value={
                "files_processed": 5,
                "files_updated": 4,
                "labels_reassigned": 3,
                "labels_dropped": 1,
                "empty_labels": 0,
                "images_removed": 0,
            },
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["medetect-yolo", "relabel", "--config", str(config_path)],
        )
        module = self._import_yolo_main(monkeypatch, relabel=mock_command)

        with patch.object(module, "append_command_history") as mock_log:
            module.main()

        mock_command.assert_called_once()
        mock_log.assert_called_once_with(
            dataset_root.resolve(),
            command="yolo relabel",
            result={
                "files_processed": 5,
                "files_updated": 4,
                "labels_reassigned": 3,
                "labels_dropped": 1,
                "empty_labels": 0,
                "images_removed": 0,
            },
        )

    def test_restore_logs_history(self, tmp_path: Path, monkeypatch) -> None:
        """restore 完了後に dataset root 直下へ履歴を追記する。"""
        dataset_root = tmp_path / "dataset"
        config_path = tmp_path / "dataset.yaml"
        self._write_dataset_yaml(dataset_root, config_path)
        module = self._import_yolo_main(monkeypatch)
        monkeypatch.setattr(
            sys,
            "argv",
            ["medetect-yolo", "restore", "--config", str(config_path)],
        )

        with (
            patch.object(module, "restore_dataset_splits") as mock_command,
            patch.object(module, "append_command_history") as mock_log,
        ):
            module.main()

        mock_command.assert_called_once()
        mock_log.assert_called_once_with(
            dataset_root.resolve(),
            command="yolo restore",
        )