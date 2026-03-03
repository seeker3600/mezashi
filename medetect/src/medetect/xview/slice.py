"""トレーニング画像を指定分解能でスライスするモジュール。

処理概要:
  - 入力: GeoTIFF 画像 (入力フォルダ/images/train) + YOLO ラベル (入力フォルダ/labels/train)
  - 各画像を指定の分解能にリサンプリングし、指定サイズのタイルに切り出す
  - 出力: PNG 画像 (出力フォルダ/images/train) + YOLO ラベル (出力フォルダ/labels/train)

使い方::

    from medetect.xview.slice import slice_training_images

    slice_training_images(
        input_dir="datasets/xView",
        output_dir="datasets/xView_sliced",
        resolution=(0.3, 10.0),
        image_size=640,
        overlap=0.2,
    )
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)


class _YoloBBox(NamedTuple):
    """YOLO 形式バウンディングボックス（正規化済み）。"""

    class_index: int
    cx: float
    cy: float
    w: float
    h: float


def _read_yolo_labels(label_path: Path) -> list[_YoloBBox]:
    """YOLO ラベルファイルを読み込む。"""
    if not label_path.exists():
        return []
    bboxes: list[_YoloBBox] = []
    with label_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                logger.warning("不正なラベル行をスキップ: %s", line)
                continue
            class_index = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
            bboxes.append(_YoloBBox(class_index, cx, cy, w, h))
    return bboxes


def _clip_bbox_to_tile(
    bbox: _YoloBBox,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    img_w: int,
    img_h: int,
) -> _YoloBBox | None:
    """タイル座標系でのバウンディングボックスを計算する。

    境界で切れたラベル（中心がタイル外）は None を返す。
    """
    # 正規化座標 -> ピクセル座標（リサンプリング後の画像全体基準）
    abs_cx = bbox.cx * img_w
    abs_cy = bbox.cy * img_h
    abs_w = bbox.w * img_w
    abs_h = bbox.h * img_h

    # 中心がタイル内にあるか
    if not (tile_x <= abs_cx < tile_x + tile_size and tile_y <= abs_cy < tile_y + tile_size):
        return None

    # タイル座標系に変換して正規化
    new_cx = (abs_cx - tile_x) / tile_size
    new_cy = (abs_cy - tile_y) / tile_size
    new_w = abs_w / tile_size
    new_h = abs_h / tile_size

    # タイル内にクランプ
    new_cx = max(0.0, min(1.0, new_cx))
    new_cy = max(0.0, min(1.0, new_cy))
    new_w = max(0.0, min(1.0, new_w))
    new_h = max(0.0, min(1.0, new_h))

    return _YoloBBox(bbox.class_index, new_cx, new_cy, new_w, new_h)


def _write_yolo_labels(label_path: Path, bboxes: list[_YoloBBox]) -> None:
    """YOLO ラベルファイルを書き込む。"""
    with label_path.open("w", encoding="utf-8") as f:
        for bbox in bboxes:
            f.write(
                f"{bbox.class_index} "
                f"{bbox.cx:.6f} {bbox.cy:.6f} "
                f"{bbox.w:.6f} {bbox.h:.6f}\n"
            )


def _get_geotiff_resolution(dataset: rasterio.DatasetReader) -> float:
    """GeoTIFF のピクセルあたり地上分解能 (メートル) を返す。

    CRS が地理座標系（度単位）の場合は中心緯度を使ってメートルへ変換する。
    投影座標系（メートル単位）の場合はそのまま平均を取る。
    """
    transform = dataset.transform
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    if dataset.crs and dataset.crs.is_geographic:
        # 度単位 → メートルへ変換
        bounds = dataset.bounds
        center_lat = (bounds.top + bounds.bottom) / 2.0
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
        res_x_m = res_x * meters_per_deg_lon
        res_y_m = res_y * meters_per_deg_lat
        return (res_x_m + res_y_m) / 2.0

    return (res_x + res_y) / 2.0


def _choose_resolution(
    resolution: float | tuple[float, float],
    *,
    max_resolution: float | None = None,
) -> float:
    """分解能を選択する。範囲指定の場合はランダム。

    Parameters
    ----------
    resolution:
        固定値または (min, max) の範囲。
    max_resolution:
        実際に使える上限値。指定時は min(選択値, max_resolution) を返す。
    """
    if isinstance(resolution, tuple):
        low, high = resolution
        if max_resolution is not None:
            high = min(high, max_resolution)
            low = min(low, high)
        return random.uniform(low, high)
    if max_resolution is not None:
        return min(resolution, max_resolution)
    return resolution


def slice_training_images(
    input_dir: str | Path,
    output_dir: str | Path,
    resolution: float | tuple[float, float],
    image_size: int,
    *,
    overlap: float = 0.0,
) -> dict[str, int]:
    """トレーニング画像を指定分解能でスライスする。

    Parameters
    ----------
    input_dir:
        入力フォルダ。``images/train`` と ``labels/train`` を含む。
    output_dir:
        出力フォルダ。``images/train`` と ``labels/train`` が生成される。
    resolution:
        ピクセルあたり分解能 (メートル)。
        float の場合は固定値、tuple の場合は (min, max) の範囲でランダム。
    image_size:
        出力タイルの一辺のピクセル数。
    overlap:
        タイル間のオーバーラップ率 (0.0〜1.0)。デフォルト 0.0。

    Returns
    -------
    dict[str, int]
        ``{"images_processed": N, "tiles_created": N, "labels_created": N}`` の統計情報。
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    images_in = input_dir / "images" / "train"
    labels_in = input_dir / "labels" / "train"
    images_out = output_dir / "images" / "train"
    labels_out = output_dir / "labels" / "train"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    step = max(1, int(image_size * (1.0 - overlap)))

    stats_images = 0
    stats_tiles = 0
    stats_labels = 0

    tif_files = sorted(images_in.glob("*.tif"))
    if not tif_files:
        logger.warning("GeoTIFF ファイルが見つかりません: %s", images_in)
        return {"images_processed": 0, "tiles_created": 0, "labels_created": 0}

    logger.info("入力画像数: %d", len(tif_files))

    for file_idx, tif_path in enumerate(tif_files):
        stem = tif_path.stem
        label_path = labels_in / f"{stem}.txt"
        bboxes = _read_yolo_labels(label_path)

        try:
            with rasterio.open(tif_path) as dataset:
                native_res = _get_geotiff_resolution(dataset)
                if native_res <= 0:
                    logger.warning("無効な分解能 (%.6f): %s", native_res, tif_path.name)
                    continue

                # image_size 未満にならない最大 target_res を算出
                min_dim = min(dataset.width, dataset.height)
                max_target_res = native_res * min_dim / image_size

                target_res = _choose_resolution(
                    resolution, max_resolution=max_target_res,
                )
                scale_factor = native_res / target_res

                # リサンプリング後のサイズ
                new_width = max(1, int(dataset.width * scale_factor))
                new_height = max(1, int(dataset.height * scale_factor))

                logger.debug(
                    "[%d/%d] %s: native=%.4f m/px, target=%.4f m/px, "
                    "scale=%.4f, resampled=%dx%d",
                    file_idx + 1, len(tif_files), tif_path.name,
                    native_res, target_res, scale_factor,
                    new_width, new_height,
                )

                if new_width < image_size or new_height < image_size:
                    logger.warning(
                        "リサンプリング後サイズ (%dx%d) が image_size (%d) 未満"
                        "のためスキップ: %s",
                        new_width, new_height, image_size, tif_path.name,
                    )
                    continue

                # 画像データの読み込み・リサンプリング
                data = dataset.read(
                    out_shape=(dataset.count, new_height, new_width),
                    resampling=Resampling.bilinear,
                )

            # (bands, height, width) -> (height, width, bands) for PIL
            if data.shape[0] == 1:
                img_array = data[0]
            else:
                img_array = np.transpose(data, (1, 2, 0))
            del data

            # タイルに切り出し
            tile_idx = 0
            for y in range(0, new_height, step):
                for x in range(0, new_width, step):
                    # タイル領域
                    x_end = x + image_size
                    y_end = y + image_size

                    if x_end > new_width or y_end > new_height:
                        continue

                    tile_data = img_array[y:y_end, x:x_end]

                    # タイル用ラベル計算
                    tile_bboxes: list[_YoloBBox] = []
                    for bbox in bboxes:
                        clipped = _clip_bbox_to_tile(
                            bbox, x, y, image_size, new_width, new_height
                        )
                        if clipped is not None:
                            tile_bboxes.append(clipped)

                    # ファイル名生成
                    tile_name = f"{stem}_{tile_idx:04d}"

                    # PNG 保存
                    tile_img = Image.fromarray(tile_data)
                    tile_img.save(images_out / f"{tile_name}.png")

                    # ラベル保存
                    _write_yolo_labels(
                        labels_out / f"{tile_name}.txt", tile_bboxes,
                    )

                    stats_tiles += 1
                    stats_labels += len(tile_bboxes)
                    tile_idx += 1

            del img_array
            stats_images += 1

            if (file_idx + 1) % 100 == 0 or file_idx + 1 == len(tif_files):
                logger.info(
                    "進捗: %d/%d 画像処理済み (タイル: %d)",
                    file_idx + 1, len(tif_files), stats_tiles,
                )

        except Exception:
            logger.exception("画像処理中にエラー: %s", tif_path.name)

    summary = {
        "images_processed": stats_images,
        "tiles_created": stats_tiles,
        "labels_created": stats_labels,
    }
    logger.info(
        "スライス完了 — 画像: %(images_processed)d, "
        "タイル: %(tiles_created)d, ラベル: %(labels_created)d",
        summary,
    )
    return summary
