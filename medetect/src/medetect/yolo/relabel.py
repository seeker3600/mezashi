"""YOLO detect ラベルのクラス ID を再マッピングする。"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import random
from pathlib import Path

from tqdm import tqdm

from medetect.yolo.dataset_yaml import load_dataset_yaml, resolve_dataset_root

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _normalize_merges(
    raw_merges: object,
) -> dict[int, int | dict[str, int | float]]:
    """YAML から読んだ merges を正規化する。

    値が整数の場合はシンプルなクラス再マッピング。
    値が辞書の場合はサイズベースの条件付き再マッピング
    (``{"threshold": float, "below": int, "above": int}``)。
    """
    if raw_merges is None:
        return {}
    if not isinstance(raw_merges, dict):
        raise TypeError("'merges' must be a mapping of source class to target class.")

    merges: dict[int, int | dict[str, int | float]] = {}
    for source, target in raw_merges.items():
        s = int(source)
        if isinstance(target, dict):
            for key in ("threshold", "below", "above"):
                if key not in target:
                    raise KeyError(f"size merge rule missing required key: '{key}'")
            merges[s] = {
                "threshold": float(target["threshold"]),
                "below": int(target["below"]),
                "above": int(target["above"]),
            }
        else:
            merges[s] = int(target)
    return merges


def _load_relabel_config(
    config_path: str | Path,
) -> tuple[Path, dict[int, int | dict[str, int | float]], float]:
    """relabel 用 YAML を読み込み、データセット root と merges と目標比率を返す。"""
    config_file, config = load_dataset_yaml(config_path)

    if "path" not in config:
        raise KeyError("Dataset YAML must define 'path'.")

    dataset_root = _resolve_dataset_root(config["path"], config_file)
    merges = _normalize_merges(config.get("merges"))
    empty_image_ratio = _normalize_probability(config.get("empty_image_ratio", 1.0))
    return dataset_root, merges, empty_image_ratio


def _resolve_dataset_root(path_value: object, config_path: Path) -> Path:
    """設定中の path を絶対パスへ解決する。"""
    return resolve_dataset_root(path_value, config_path)


def _normalize_probability(value: object) -> float:
    """0.0 から 1.0 の保持確率へ正規化する。"""
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("empty_image_ratio must be between 0.0 and 1.0.")
    return probability


def _compute_geo_resolution(
    res_x: float,
    res_y: float,
    *,
    is_geographic: bool,
    center_lat: float = 0.0,
) -> float:
    """ピクセルあたり地上分解能 (メートル) を計算する。

    ``medetect.xview.slice._compute_geo_resolution`` と同じロジック。
    """
    if is_geographic:
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
        return (res_x * meters_per_deg_lon + res_y * meters_per_deg_lat) / 2.0
    return (res_x + res_y) / 2.0


def _get_image_geo_info(
    label_path: Path,
    dataset_root: Path,
) -> tuple[int, int, float] | None:
    """ラベルに対応する GeoTIFF 画像の (幅, 高さ, 分解能 m/px) を取得する。

    GeoTIFF が見つからない場合は ``None`` を返す。
    """
    import rasterio

    try:
        relative_label = label_path.relative_to(dataset_root / "labels")
    except ValueError:
        return None

    image_dir = dataset_root / "images" / relative_label.parent
    stem = label_path.stem

    for ext in (".tif", ".tiff"):
        image_path = image_dir / f"{stem}{ext}"
        if image_path.exists():
            with rasterio.open(image_path) as ds:
                transform = ds.transform
                res_x = abs(transform.a)
                res_y = abs(transform.e)
                is_geographic = bool(ds.crs and ds.crs.is_geographic)
                center_lat = 0.0
                if is_geographic:
                    bounds = ds.bounds
                    center_lat = (bounds.top + bounds.bottom) / 2.0
                resolution = _compute_geo_resolution(
                    res_x, res_y,
                    is_geographic=is_geographic,
                    center_lat=center_lat,
                )
                return ds.width, ds.height, resolution
    return None


def _relabel_line(
    line: str,
    merges: dict[int, int | dict[str, int | float]],
    img_width: int = 0,
    img_height: int = 0,
    resolution: float = 0.0,
) -> tuple[str | None, bool, bool]:
    """1 行分の YOLO detect ラベルを再マッピングする。"""
    stripped = line.strip()
    if not stripped:
        return None, False, False

    tokens = stripped.split()
    if len(tokens) < 5:
        raise ValueError(f"Invalid YOLO detect label line: {line.rstrip()}")

    source_class = int(tokens[0])
    rule = merges.get(source_class, source_class)

    if isinstance(rule, dict):
        if img_width > 0 and img_height > 0 and resolution > 0:
            w_meters = float(tokens[3]) * img_width * resolution
            h_meters = float(tokens[4]) * img_height * resolution
            longest = max(w_meters, h_meters)
            if longest < float(rule["threshold"]):
                target_class = int(rule["below"])
            else:
                target_class = int(rule["above"])
        else:
            target_class = source_class  # geo info なし → サイズルールをスキップ
    else:
        target_class = rule

    if target_class == -1:
        return None, False, True

    changed = target_class != source_class
    tokens[0] = str(target_class)
    return " ".join(tokens), changed, False


def _relabel_file(
    label_path: Path,
    merges: dict[int, int | dict[str, int | float]],
    dataset_root: Path | None = None,
) -> dict[str, int]:
    """単一ラベルファイルを上書き更新する。"""
    output_lines: list[str] = []
    labels_reassigned = 0
    labels_dropped = 0

    # サイズベースのルールがある場合のみ GeoTIFF から地理情報を取得する
    has_size_rules = any(isinstance(v, dict) for v in merges.values())
    img_width, img_height, resolution = 0, 0, 0.0
    if has_size_rules and dataset_root is not None:
        geo_info = _get_image_geo_info(label_path, dataset_root)
        if geo_info is not None:
            img_width, img_height, resolution = geo_info

    old_text = label_path.read_text(encoding="utf-8")
    for line in old_text.splitlines():
        new_line, changed, dropped = _relabel_line(
            line, merges,
            img_width=img_width, img_height=img_height,
            resolution=resolution,
        )
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

    if old_text != new_text:
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
    merges: dict,
    *,
    empty_image_ratio: float = 1.0,
    max_workers: int | None = None,
) -> dict[str, int]:
    """YOLO detect データセット配下の labels を指定マッピングで再ラベルする。

    empty_image_ratio は全画像に対するラベル無し画像の目標比率。
    再ラベル後の比率が目標を超えている場合、目標比率になるまでランダムに削除する。
    すでに目標以下であれば何も削除しない。
    0.0: ラベル無し画像をすべて削除 / 1.0: 削除しない

    merges の値には整数（シンプルなクラス再マッピング）と辞書
    ``{"threshold": float, "below": int, "above": int}``（サイズベースの再マッピング）
    を混在させることができる。辞書の場合は bbox の長辺がメートル単位で
    threshold 未満なら below クラスへ、以上なら above クラスへ再マッピングする。
    -1 を指定するとドロップする。
    """
    dataset_root = Path(dataset_root).resolve()
    merges = _normalize_merges(merges)
    empty_ratio = _normalize_probability(empty_image_ratio)
    labels_root = dataset_root / "labels"

    if max_workers is None:
        max_workers = os.cpu_count() or 1

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
    label_paths = sorted(labels_root.rglob("*.txt"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_relabel_file, lp, merges, dataset_root): lp
            for lp in label_paths
        }

        with tqdm(
            total=len(label_paths),
            desc="relabel",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                label_path = futures[future]
                file_stats = future.result()
                stats["files_processed"] += 1
                stats["files_updated"] += file_stats["updated"]
                stats["labels_reassigned"] += file_stats["labels_reassigned"]
                stats["labels_dropped"] += file_stats["labels_dropped"]
                stats["empty_labels"] += file_stats["is_empty"]

                if file_stats["is_empty"]:
                    empty_label_paths.append(label_path)

                pbar.update(1)

    # 目標比率に合わせて残すラベル無し画像数を計算する。
    # 目標: E_kept / (n_labeled + E_kept) = empty_ratio
    # => E_kept = empty_ratio * n_labeled / (1 - empty_ratio)
    empty_label_paths.sort()
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
    empty_image_ratio: float | None = None,
    max_workers: int | None = None,
) -> dict[str, int]:
    """YOLO detect データセット YAML を読み込み、labels を再ラベルする。"""
    dataset_root, merges, config_ratio = _load_relabel_config(config_path)
    if empty_image_ratio is None:
        empty_image_ratio = config_ratio
    return relabel_yolo_detect_labels(
        dataset_root=dataset_root,
        merges=merges,
        empty_image_ratio=empty_image_ratio,
        max_workers=max_workers,
    )