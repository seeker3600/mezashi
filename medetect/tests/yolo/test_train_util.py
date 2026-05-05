from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from medetect.yolo.train_util import copy_training_artifacts


def _make_trainer(save_dir: Path, data: str) -> SimpleNamespace:
    return SimpleNamespace(
        save_dir=save_dir,
        args=SimpleNamespace(data=data),
    )


class TestCopyTrainingArtifacts:
    """copy_training_artifacts のテスト。"""

    def test_copies_dataset_command_history_and_train_script(
        self,
        tmp_path: Path,
    ) -> None:
        """dataset に履歴ログがある場合、履歴ログと train.py を run_dir へコピーする。"""
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
        run_dir = tmp_path / "runs" / "train1"
        run_dir.mkdir(parents=True)
        trainer = _make_trainer(run_dir, str(config_path))

        copy_training_artifacts(trainer)

        assert (run_dir / "command_history.jsonl").read_text(encoding="utf-8") == '{"command":"datagen"}\n'
        assert (run_dir / "train.py").exists()

    def test_skips_missing_dataset_command_history(
        self,
        tmp_path: Path,
    ) -> None:
        """dataset に履歴ログが無くても train.py のコピーは行う。"""
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        config_path = tmp_path / "dataset.yaml"
        config_path.write_text(
            f"path: {dataset_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n",
            encoding="utf-8",
        )
        run_dir = tmp_path / "runs" / "train2"
        run_dir.mkdir(parents=True)
        trainer = _make_trainer(run_dir, str(config_path))

        copy_training_artifacts(trainer)

        assert not (run_dir / "command_history.jsonl").exists()
        assert (run_dir / "train.py").exists()
