from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from medetect.yolo.train import train_kwargs, train_yolo_model


class TestTrainYoloModel:
    """train_yolo_model のテスト。"""

    def test_copies_dataset_command_history_and_train_script(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """学習成功後に dataset の履歴ログと train.py を save_dir へコピーする。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        dataset_log = dataset_root / "command_history.jsonl"
        dataset_log.write_text('{"command":"datagen"}\n', encoding="utf-8")
        config_path = tmp_path / "dataset.yaml"
        config_path.write_text(
            f"path: {dataset_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n",
            encoding="utf-8",
        )
        run_dir = tmp_path / "runs" / "detect" / "train42"
        run_dir.mkdir(parents=True)
        monkeypatch.setitem(train_kwargs, "data", str(config_path))

        with patch("medetect.yolo.train.YOLO") as mock_yolo:
            mock_yolo.return_value.train.return_value = SimpleNamespace(save_dir=run_dir)
            train_yolo_model()

        mock_yolo.return_value.train.assert_called_once()
        assert (run_dir / "command_history.jsonl").read_text(encoding="utf-8") == '{"command":"datagen"}\n'
        assert (run_dir / "train.py").exists()

    def test_skips_missing_dataset_command_history(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """dataset 側に履歴ログが無くても train.py のコピーだけは行う。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        config_path = tmp_path / "dataset.yaml"
        config_path.write_text(
            f"path: {dataset_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n",
            encoding="utf-8",
        )
        run_dir = tmp_path / "runs" / "detect" / "train43"
        run_dir.mkdir(parents=True)
        monkeypatch.setitem(train_kwargs, "data", str(config_path))

        with patch("medetect.yolo.train.YOLO") as mock_yolo:
            mock_yolo.return_value.train.return_value = SimpleNamespace(save_dir=run_dir)
            train_yolo_model()

        mock_yolo.return_value.train.assert_called_once()
        assert not (run_dir / "command_history.jsonl").exists()
        assert (run_dir / "train.py").exists()


def test_cli_train_calls_train_yolo_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """train サブコマンドは train_yolo_model を引数なしで呼ぶ。"""
    monkeypatch.setattr(sys, "argv", ["python", "train"])
    module = importlib.import_module("medetect.yolo.__main__")
    with patch.object(module, "train_yolo_model") as mock_train:
        module.main()
    mock_train.assert_called_once_with()
