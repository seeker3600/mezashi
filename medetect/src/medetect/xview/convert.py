"""xView GeoJSON トレーニングラベルを YOLO フォーマットに変換するモジュール。

変換仕様:
  - 入力: xView_train.geojson（各 Feature は bounds_imcoords / type_id / image_id を持つ）
  - 出力: 画像ごとに "<image_stem>.txt" を生成（YOLO 形式: class cx cy w h、正規化済み）
  - 合わせて classes.txt も出力

使い方::

    from medetect.xview.convert import convert_xview_to_yolo

    convert_xview_to_yolo(
        geojson_path="datasets/xView/train_labels/xView_train.geojson",
        images_dir="datasets/xView/train_images/train_images",
        output_dir="datasets/xView/labels/train",
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

from PIL import Image

from medetect.xview.classes import (
    XVIEW_CLASS_NAMES,
    XVIEW_TYPE_ID_TO_INDEX,
)

logger = logging.getLogger(__name__)


class BBox(NamedTuple):
    """正規化済み YOLO バウンディングボックス。"""

    class_index: int
    cx: float
    cy: float
    w: float
    h: float


def _parse_bounds(bounds_str: str) -> tuple[int, int, int, int]:
    """bounds_imcoords 文字列 "x_min,y_min,x_max,y_max" をパース。"""
    parts = bounds_str.split(",")
    if len(parts) != 4:
        raise ValueError(f"不正な bounds_imcoords 形式: {bounds_str!r}")
    x_min, y_min, x_max, y_max = (int(p.strip()) for p in parts)
    return x_min, y_min, x_max, y_max


def _to_yolo_bbox(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    img_w: int,
    img_h: int,
    class_index: int,
) -> BBox:
    """ピクセル座標を正規化 YOLO 形式に変換。"""
    cx = ((x_min + x_max) / 2.0) / img_w
    cy = ((y_min + y_max) / 2.0) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    # 画像範囲内にクランプ
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return BBox(class_index, cx, cy, w, h)


def _get_image_size(image_path: Path) -> tuple[int, int]:
    """画像のサイズ (width, height) をヘッダのみ読み込んで取得。"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def convert_xview_to_yolo(
    geojson_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    skip_unknown_type_ids: bool = True,
    skip_missing_images: bool = True,
) -> dict[str, int]:
    """xView GeoJSON を YOLO ラベルファイル群に変換する。

    Parameters
    ----------
    geojson_path:
        xView_train.geojson のパス。
    images_dir:
        画像ファイル（.tif）が置かれているディレクトリ。
    output_dir:
        YOLO ラベル（.txt）と classes.txt の出力先ディレクトリ。
    skip_unknown_type_ids:
        True の場合、未知の type_id を持つ Feature をスキップ（警告のみ）。
        False の場合 KeyError が発生。
    skip_missing_images:
        True の場合、対応する画像が見つからない image_id をスキップ（警告のみ）。
        False の場合 FileNotFoundError が発生。

    Returns
    -------
    dict[str, int]
        ``{"written": N, "skipped_type_id": N, "skipped_image": N,
            "images": N}`` の統計情報。
    """
    geojson_path = Path(geojson_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("GeoJSON 読込: %s", geojson_path)
    with geojson_path.open(encoding="utf-8") as f:
        geojson = json.load(f)

    features: list[dict] = geojson.get("features", [])
    logger.info("Feature 総数: %d", len(features))

    # --- image_id ごとに Feature をグルーピング ---
    by_image: dict[str, list[dict]] = {}
    stats_skipped_type: int = 0

    for feat in features:
        props: dict = feat.get("properties", {})
        type_id: int | None = props.get("type_id")
        image_id: str | None = props.get("image_id")

        if image_id is None:
            logger.warning("image_id が存在しない Feature をスキップ: %s", props)
            continue

        if type_id not in XVIEW_TYPE_ID_TO_INDEX:
            if skip_unknown_type_ids:
                logger.debug(
                    "未知の type_id %s をスキップ (image: %s)", type_id, image_id
                )
                stats_skipped_type += 1
                continue
            raise KeyError(f"未知の type_id: {type_id}（image_id: {image_id}）")

        by_image.setdefault(image_id, []).append(feat)

    # --- 画像ごとにラベルファイルを生成 ---
    stats_written: int = 0
    stats_skipped_image: int = 0
    stats_images: int = 0

    for image_id, image_features in by_image.items():
        image_path = images_dir / image_id
        if not image_path.exists():
            if skip_missing_images:
                logger.warning("画像が見つかりません。スキップ: %s", image_path)
                stats_skipped_image += len(image_features)
                continue
            raise FileNotFoundError(f"画像が見つかりません: {image_path}")

        try:
            img_w, img_h = _get_image_size(image_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("画像サイズ取得失敗 (%s): %s", image_path.name, exc)
            stats_skipped_image += len(image_features)
            continue

        bboxes: list[BBox] = []
        for feat in image_features:
            props = feat["properties"]
            type_id = props["type_id"]
            bounds_str: str = props.get("bounds_imcoords", "")
            try:
                x_min, y_min, x_max, y_max = _parse_bounds(bounds_str)
            except ValueError as exc:
                logger.warning("bounds_imcoords パース失敗: %s", exc)
                continue

            # 0 面積のボックスはスキップ
            if x_min >= x_max or y_min >= y_max:
                logger.debug(
                    "面積 0 のボックスをスキップ: %s (image: %s)", bounds_str, image_id
                )
                continue

            class_index = XVIEW_TYPE_ID_TO_INDEX[type_id]
            bbox = _to_yolo_bbox(x_min, y_min, x_max, y_max, img_w, img_h, class_index)
            bboxes.append(bbox)

        # ラベルファイル出力（元の .tif 拡張子を .txt に）
        stem = Path(image_id).stem
        label_path = output_dir / f"{stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            for bbox in bboxes:
                f.write(
                    f"{bbox.class_index} "
                    f"{bbox.cx:.6f} {bbox.cy:.6f} "
                    f"{bbox.w:.6f} {bbox.h:.6f}\n"
                )

        stats_written += len(bboxes)
        stats_images += 1
        logger.debug("書込完了: %s (%d 件)", label_path.name, len(bboxes))

    # classes.txt を出力
    classes_path = output_dir / "classes.txt"
    with classes_path.open("w", encoding="utf-8") as f:
        for name in XVIEW_CLASS_NAMES:
            f.write(f"{name}\n")
    logger.info("classes.txt 書込: %s", classes_path)

    summary = {
        "written": stats_written,
        "skipped_type_id": stats_skipped_type,
        "skipped_image": stats_skipped_image,
        "images": stats_images,
    }
    logger.info(
        "変換完了 — 画像: %(images)d, ラベル: %(written)d 件書込, "
        "type_id スキップ: %(skipped_type_id)d, 画像スキップ: %(skipped_image)d",
        summary,
    )
    return summary
