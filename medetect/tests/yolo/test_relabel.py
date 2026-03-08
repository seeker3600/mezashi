from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from medetect.yolo.__main__ import main
from medetect.yolo.relabel import relabel_yolo_detect_dataset


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
    }
    assert (labels_dir / "sample.txt").read_text(encoding="utf-8") == "1 0.5 0.5 0.2 0.2\n"


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