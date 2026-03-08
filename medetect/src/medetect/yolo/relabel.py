"""YOLO detect ラベルのクラス ID を再マッピングする。"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _normalize_merges(raw_merges: object) -> dict[int, int]:
    """YAML から読んだ merges を int -> int の辞書へ正規化する。"""
    if raw_merges is None:
        return {}
    if not isinstance(raw_merges, dict):
        raise TypeError("'merges' must be a mapping of source class to target class.")

    merges: dict[int, int] = {}
    for source, target in raw_merges.items():
        merges[int(source)] = int(target)
    return merges


def _load_relabel_config(config_path: str | Path) -> tuple[Path, dict[int, int]]:
    """relabel 用 YAML を読み込み、データセット root と merges を返す。"""
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError("Dataset YAML must contain a top-level mapping.")
    if "path" not in config:
        raise KeyError("Dataset YAML must define 'path'.")

    dataset_root = _resolve_dataset_root(config["path"], config_file)
    merges = _normalize_merges(config.get("merges"))
    return dataset_root, merges


def _resolve_dataset_root(path_value: object, config_path: Path) -> Path:
    """設定中の path を絶対パスへ解決する。"""
    if not isinstance(path_value, (str, Path)):
        raise TypeError("'path' must be a string or path-like value.")

    dataset_path = Path(path_value)
    if dataset_path.is_absolute():
        return dataset_path.resolve()

    config_relative = (config_path.parent / dataset_path).resolve()
    if config_relative.exists():
        return config_relative

    try:
        from ultralytics import settings
    except ImportError:
        return config_relative

    datasets_root = Path(settings["datasets_dir"]).expanduser()
    ultralytics_relative = (datasets_root / dataset_path).resolve()
    if ultralytics_relative.exists():
        return ultralytics_relative

    return ultralytics_relative


def _relabel_line(line: str, merges: dict[int, int]) -> tuple[str | None, bool, bool]:
    """1 行分の YOLO detect ラベルを再マッピングする。"""
    stripped = line.strip()
    if not stripped:
        return None, False, False

    tokens = stripped.split()
    if len(tokens) < 5:
        raise ValueError(f"Invalid YOLO detect label line: {line.rstrip()}")

    source_class = int(tokens[0])
    target_class = merges.get(source_class, source_class)
    if target_class == -1:
        return None, False, True

    changed = target_class != source_class
    tokens[0] = str(target_class)
    return " ".join(tokens), changed, False


def _relabel_file(label_path: Path, merges: dict[int, int]) -> dict[str, int]:
    """単一ラベルファイルを上書き更新する。"""
    output_lines: list[str] = []
    labels_reassigned = 0
    labels_dropped = 0

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            new_line, changed, dropped = _relabel_line(line, merges)
            if dropped:
                labels_dropped += 1
                continue
            if new_line is None:
                continue
            if changed:
                labels_reassigned += 1
            output_lines.append(new_line)

    new_text = "\n".join(output_lines)
    if output_lines:
        new_text += "\n"

    old_text = label_path.read_text(encoding="utf-8")
    label_path.write_text(new_text, encoding="utf-8")

    return {
        "updated": int(old_text != new_text),
        "labels_reassigned": labels_reassigned,
        "labels_dropped": labels_dropped,
    }


def relabel_yolo_detect_dataset(config_path: str | Path) -> dict[str, int]:
    """YOLO detect データセット配下の labels を merges に従って再ラベルする。"""
    dataset_root, merges = _load_relabel_config(config_path)
    labels_root = dataset_root / "labels"

    if not labels_root.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_root}")

    stats = {
        "files_processed": 0,
        "files_updated": 0,
        "labels_reassigned": 0,
        "labels_dropped": 0,
    }

    for label_path in sorted(labels_root.rglob("*.txt")):
        file_stats = _relabel_file(label_path, merges)
        stats["files_processed"] += 1
        stats["files_updated"] += file_stats["updated"]
        stats["labels_reassigned"] += file_stats["labels_reassigned"]
        stats["labels_dropped"] += file_stats["labels_dropped"]

    logger.info(
        "relabelled %d files under %s",
        stats["files_processed"],
        labels_root,
    )
    return stats