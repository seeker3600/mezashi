from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from medetect.command_history import append_command_history, command_history_path


class TestCommandHistoryPath:
    """command_history_path のテスト。"""

    def test_returns_log_path_under_dataset_root(self, tmp_path: Path) -> None:
        """dataset root 直下の JSONL パスを返す。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        result = command_history_path(dataset_root)

        assert result == dataset_root.resolve() / "command_history.jsonl"


class TestAppendCommandHistory:
    """append_command_history のテスト。"""

    def test_appends_one_json_record_per_execution(self, tmp_path: Path) -> None:
        """複数回呼ぶと JSONL に1行ずつ追記される。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        append_command_history(
            dataset_root,
            command="datagen",
            argv=["medetect-datagen", "--count", "10"],
            cwd=tmp_path,
            result={"images": 10, "ships": 7},
        )
        append_command_history(
            dataset_root,
            command="expand-obb",
            argv=["medetect-yolo", "expand-obb", "--config", "dataset.yaml"],
            cwd=tmp_path,
        )

        lines = (dataset_root / "command_history.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        first_record = json.loads(lines[0])
        second_record = json.loads(lines[1])
        assert first_record["command"] == "datagen"
        assert first_record["argv"] == ["medetect-datagen", "--count", "10"]
        assert first_record["cwd"] == str(tmp_path.resolve())
        assert first_record["dataset_root"] == str(dataset_root.resolve())
        assert first_record["status"] == "success"
        assert first_record["result"] == {"images": 10, "ships": 7}
        assert second_record["command"] == "expand-obb"
        assert second_record["argv"] == ["medetect-yolo", "expand-obb", "--config", "dataset.yaml"]
        assert second_record["status"] == "success"

    def test_uses_process_defaults_for_argv_and_cwd(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """argv と cwd を省略したときは現在プロセスの値を使う。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["medetect-yolo", "valsplit", "--fraction", "0.2"])

        append_command_history(dataset_root, command="valsplit")

        record = json.loads((dataset_root / "command_history.jsonl").read_text(encoding="utf-8"))
        assert record["command"] == "valsplit"
        assert record["argv"] == ["medetect-yolo", "valsplit", "--fraction", "0.2"]
        assert record["cwd"] == str(tmp_path.resolve())
        assert record["dataset_root"] == str(dataset_root.resolve())
        assert record["status"] == "success"