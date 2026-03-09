from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import pytest

from medetect.yolo.__main__ import main
from medetect.yolo.relabel import relabel_yolo_detect_dataset, relabel_yolo_detect_labels


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

    monkeypatch.setattr(random, "random", lambda: 0.5)

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