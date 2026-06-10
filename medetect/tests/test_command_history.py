from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

from medetect.command_history import append_command_history, command_history_path, read_command_history


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
        """複数回呼ぶと JSONL に1レコードずつ追記される。"""
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

        records = read_command_history(dataset_root)
        assert len(records) == 2

        first_record = records[0]
        second_record = records[1]
        assert first_record["command"] == "datagen"
        assert first_record["argv"] == ["medetect-datagen", "--count", "10"]
        assert first_record["cwd"] == str(tmp_path.resolve())
        assert first_record["dataset_root"] == str(dataset_root.resolve())
        assert first_record["status"] == "success"
        assert first_record["result"] == {"images": 10, "ships": 7}
        assert re.fullmatch(r"[0-9a-f]{40}", first_record["git_commit_hash"] or "")
        assert second_record["command"] == "expand-obb"
        assert second_record["argv"] == ["medetect-yolo", "expand-obb", "--config", "dataset.yaml"]
        assert second_record["status"] == "success"
        assert re.fullmatch(r"[0-9a-f]{40}", second_record["git_commit_hash"] or "")

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

        records = read_command_history(dataset_root)
        assert len(records) == 1
        record = records[0]
        assert record["command"] == "valsplit"
        assert record["argv"] == ["medetect-yolo", "valsplit", "--fraction", "0.2"]
        assert record["cwd"] == str(tmp_path.resolve())
        assert record["dataset_root"] == str(dataset_root.resolve())
        assert record["status"] == "success"
        assert re.fullmatch(r"[0-9a-f]{40}", record["git_commit_hash"] or "")

    def test_records_null_git_hash_when_lookup_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Git ハッシュ取得失敗時は null を記録する。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        monkeypatch.setattr("medetect.command_history._git_commit_hash", lambda: None)

        append_command_history(dataset_root, command="valsplit", argv=["medetect-yolo", "valsplit"])

        records = read_command_history(dataset_root)
        assert len(records) == 1
        assert "git_commit_hash" in records[0]
        assert records[0]["git_commit_hash"] is None

    def test_record_keys_are_sorted_and_indented(self, tmp_path: Path) -> None:
        """キーがアルファベット順・インデント付きで書き込まれる。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        append_command_history(
            dataset_root,
            command="datagen",
            argv=["medetect-datagen"],
            cwd=tmp_path,
            result={"ships": 3, "images": 2},
        )

        raw = (dataset_root / "command_history.jsonl").read_text(encoding="utf-8")
        # インデント付き（複数行）であること
        assert "\n  " in raw
        # json.loads はキーの挿入順を保持するので sort_keys の効果が確認できる
        record = read_command_history(dataset_root)[0]
        top_keys = list(record.keys())
        assert top_keys == sorted(top_keys)
        result_keys = list(record["result"].keys())
        assert result_keys == sorted(result_keys)


class TestOverwriteCommandHistory:
    """overwrite=True のときに既存ログをリセットするテスト。"""

    def test_overwrite_replaces_previous_history(self, tmp_path: Path) -> None:
        """overwrite=True で呼ぶと既存エントリが消えて新しい1件のみになる。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        append_command_history(dataset_root, command="datagen", argv=[], cwd=tmp_path)
        append_command_history(dataset_root, command="expand-obb", argv=[], cwd=tmp_path)

        # overwrite=True で上書き
        append_command_history(
            dataset_root,
            command="datagen",
            argv=["medetect-datagen", "--count", "5"],
            cwd=tmp_path,
            result={"images": 5, "ships": 3},
            overwrite=True,
        )

        records = read_command_history(dataset_root)
        assert len(records) == 1
        assert records[0]["command"] == "datagen"
        assert records[0]["result"] == {"images": 5, "ships": 3}

    def test_overwrite_false_still_appends(self, tmp_path: Path) -> None:
        """overwrite=False (デフォルト) では追記される。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        append_command_history(dataset_root, command="datagen", argv=[], cwd=tmp_path)
        append_command_history(
            dataset_root, command="expand-obb", argv=[], cwd=tmp_path, overwrite=False
        )

        records = read_command_history(dataset_root)
        assert len(records) == 2


class TestReadCommandHistory:
    """read_command_history のテスト。"""

    def test_returns_empty_list_when_no_file(self, tmp_path: Path) -> None:
        """ファイルが存在しない場合は空リストを返す。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()

        assert read_command_history(dataset_root) == []