"""YOLO detect ラベルのクラス ID を再マッピングする。"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


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


def _load_relabel_config(config_path: str | Path) -> tuple[Path, dict[int, int], float]:
    """relabel 用 YAML を読み込み、データセット root と merges と確率を返す。"""
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError("Dataset YAML must contain a top-level mapping.")
    if "path" not in config:
        raise KeyError("Dataset YAML must define 'path'.")

    dataset_root = _resolve_dataset_root(config["path"], config_file)
    merges = _normalize_merges(config.get("merges"))
    empty_image_keep_prob = _normalize_probability(config.get("empty_image_keep_prob", 1.0))
    return dataset_root, merges, empty_image_keep_prob


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


def _normalize_probability(value: object) -> float:
    """0.0 から 1.0 の保持確率へ正規化する。"""
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("empty_image_keep_prob must be between 0.0 and 1.0.")
    return probability


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
        "is_empty": int(not output_lines),
    }


def _remove_image_and_label(label_path: Path, dataset_root: Path) -> int:
    """空ラベルになったサンプルのラベルと対応画像を削除する。"""
    removed_images = 0
    label_path.unlink(missing_ok=True)

    try:
        relative_label = label_path.relative_to(dataset_root / "labels")
    except ValueError:
        relative_label = Path(label_path.name)

    image_dir = dataset_root / "images" / relative_label.parent
    stem = label_path.stem
    for extension in _IMAGE_EXTENSIONS:
        image_path = image_dir / f"{stem}{extension}"
        if image_path.exists():
            image_path.unlink()
            removed_images += 1

    return removed_images


def relabel_yolo_detect_labels(
    dataset_root: str | Path,
    merges: dict[int, int] | dict[str, int] | dict[int, str] | dict[str, str],
    *,
    empty_image_keep_prob: float = 1.0,
) -> dict[str, int]:
    """YOLO detect データセット配下の labels を指定マッピングで再ラベルする。

    empty_image_keep_prob は全画像に対するラベル無し画像の目標比率。
    再ラベル後の比率が目標を超えている場合、目標比率になるまでランダムに削除する。
    すでに目標以下であれば何も削除しない。
    0.0: ラベル無し画像をすべて削除 / 1.0: 削除しない
    """
    dataset_root = Path(dataset_root).resolve()
    merges = _normalize_merges(merges)
    empty_ratio = _normalize_probability(empty_image_keep_prob)
    labels_root = dataset_root / "labels"

    if not labels_root.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_root}")

    stats = {
        "files_processed": 0,
        "files_updated": 0,
        "labels_reassigned": 0,
        "labels_dropped": 0,
        "empty_labels": 0,
        "images_removed": 0,
    }

    empty_label_paths: list[Path] = []

    for label_path in sorted(labels_root.rglob("*.txt")):
        file_stats = _relabel_file(label_path, merges)
        stats["files_processed"] += 1
        stats["files_updated"] += file_stats["updated"]
        stats["labels_reassigned"] += file_stats["labels_reassigned"]
        stats["labels_dropped"] += file_stats["labels_dropped"]
        stats["empty_labels"] += file_stats["is_empty"]

        if file_stats["is_empty"]:
            empty_label_paths.append(label_path)

    # 目標比率に合わせて残すラベル無し画像数を計算する。
    # 目標: E_kept / (n_labeled + E_kept) = empty_ratio
    # => E_kept = empty_ratio * n_labeled / (1 - empty_ratio)
    n_empty = len(empty_label_paths)
    n_labeled = stats["files_processed"] - n_empty

    if empty_ratio >= 1.0:
        keep_count = n_empty
    elif empty_ratio <= 0.0:
        keep_count = 0
    else:
        keep_count = min(n_empty, int(empty_ratio * n_labeled / (1.0 - empty_ratio)))

    n_to_remove = n_empty - keep_count
    if n_to_remove > 0:
        for label_path in random.sample(empty_label_paths, n_to_remove):
            stats["images_removed"] += _remove_image_and_label(label_path, dataset_root)

    logger.info(
        "relabel complete: files=%d updated=%d relabeled=%d dropped=%d empty=%d images_removed=%d",
        stats["files_processed"],
        stats["files_updated"],
        stats["labels_reassigned"],
        stats["labels_dropped"],
        stats["empty_labels"],
        stats["images_removed"],
    )

    return stats


def relabel_yolo_detect_dataset(
    config_path: str | Path,
    *,
    empty_image_keep_prob: float | None = None,
) -> dict[str, int]:
    """YOLO detect データセット YAML を読み込み、labels を再ラベルする。"""
    dataset_root, merges, config_keep_prob = _load_relabel_config(config_path)
    if empty_image_keep_prob is None:
        empty_image_keep_prob = config_keep_prob
    return relabel_yolo_detect_labels(
        dataset_root=dataset_root,
        merges=merges,
        empty_image_keep_prob=empty_image_keep_prob,
    )