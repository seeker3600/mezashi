"""トレーニング画像を指定分解能でスライスするモジュール。

処理概要:
  - 入力: GeoTIFF 画像 (入力フォルダ/images/train) + YOLO ラベル (入力フォルダ/labels/train)
  - 各画像を rasterio でウィンドウ単位で読み出し、指定分解能にリサンプリングしてタイル化
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

import concurrent.futures
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 純粋計算ヘルパー（テスト容易化のため I/O から分離）
# ---------------------------------------------------------------------------

def _compute_geo_resolution(
    res_x: float,
    res_y: float,
    *,
    is_geographic: bool,
    center_lat: float = 0.0,
) -> float:
    """ピクセルあたり地上分解能 (メートル) を計算する。

    Parameters
    ----------
    res_x:
        X 方向のピクセルサイズ（正値）。
    res_y:
        Y 方向のピクセルサイズ（正値）。
    is_geographic:
        ``True`` の場合は度単位として中心緯度を使いメートルへ変換する。
        ``False`` の場合は既にメートル単位とみなしてそのまま平均を返す。
    center_lat:
        地理座標系の場合に使用する中心緯度（度）。デフォルト 0.0。
    """
    if is_geographic:
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
        return (res_x * meters_per_deg_lon + res_y * meters_per_deg_lat) / 2.0
    return (res_x + res_y) / 2.0


def _compute_native_tile_size(
    native_res: float,
    src_width: int,
    src_height: int,
    image_size: int,
    resolution: float | tuple[float, float],
) -> tuple[float, float] | None:
    """タイルウィンドウの ``(target_res, native_tile_size)`` を計算する。

    1 タイルが元画像のネイティブピクセルで何ピクセルに相当するかを返す。
    画像が小さすぎて 1 タイルも取れない場合は ``None`` を返す。

    Parameters
    ----------
    native_res:
        元画像のピクセルあたり地上分解能 (メートル)。
    src_width:
        元画像の幅 (ピクセル)。
    src_height:
        元画像の高さ (ピクセル)。
    image_size:
        出力タイルの一辺のピクセル数。
    resolution:
        目標分解能 (メートル)。float か (min, max) の範囲タプル。
    """
    min_dim = min(src_width, src_height)
    max_target_res = native_res * min_dim / image_size

    target_res = _choose_resolution(resolution, max_resolution=max_target_res)
    native_tile_size = image_size * target_res / native_res

    if native_tile_size > min_dim:
        return None

    return target_res, native_tile_size


def _iter_tile_windows(
    src_width: int,
    src_height: int,
    native_tile_size: float,
    overlap: float,
) -> list[tuple[int, int, float, float]]:
    """タイルウィンドウの位置を列挙する。

    Parameters
    ----------
    src_width:
        元画像の幅 (ピクセル)。
    src_height:
        元画像の高さ (ピクセル)。
    native_tile_size:
        1 タイルの一辺 (ネイティブピクセル)。
    overlap:
        タイル間のオーバーラップ率 (0.0〜1.0)。

    Returns
    -------
    list[tuple[int, int, float, float]]
        ``(tile_row, tile_col, col_off, row_off)`` のリスト。
    """
    if native_tile_size > src_width or native_tile_size > src_height:
        return []

    stride = native_tile_size * (1.0 - overlap)
    if stride <= 0:
        stride = native_tile_size  # fallback: overlap=1.0 → no overlap

    n_cols = max(1, math.floor((src_width - native_tile_size) / stride) + 1)
    n_rows = max(1, math.floor((src_height - native_tile_size) / stride) + 1)

    tiles: list[tuple[int, int, float, float]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            col_off = col * stride
            row_off = row * stride
            # Clamp so the tile does not exceed image bounds
            col_off = min(col_off, max(0.0, src_width - native_tile_size))
            row_off = min(row_off, max(0.0, src_height - native_tile_size))
            tiles.append((row, col, col_off, row_off))

    return tiles


def _compute_tile_transform(
    src_transform: Affine,
    col_off: float,
    row_off: float,
    native_tile_size: float,
    image_size: int,
) -> Affine:
    """タイルの Affine transform を計算する。

    ソース画像から切り出した (col_off, row_off) 位置のタイルが
    image_size ピクセルにリサンプリングされた場合の地理参照を返す。
    """
    scale = native_tile_size / image_size
    tile_x = (
        src_transform.c
        + col_off * src_transform.a
        + row_off * src_transform.b
    )
    tile_y = (
        src_transform.f
        + col_off * src_transform.d
        + row_off * src_transform.e
    )
    return Affine(
        src_transform.a * scale,
        src_transform.b * scale,
        tile_x,
        src_transform.d * scale,
        src_transform.e * scale,
        tile_y,
    )


def _parse_yolo_labels(
    label_path: Path,
) -> list[tuple[int, float, float, float, float]]:
    """YOLO 形式のラベルファイルを解析する。

    Returns
    -------
    list[tuple[int, float, float, float, float]]
        ``(class_id, x_center, y_center, width, height)`` のリスト。
        座標は画像サイズで正規化済み (0‑1)。
    """
    labels: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        labels.append((cls_id, xc, yc, w, h))
    return labels


def _clip_labels_to_window(
    labels: list[tuple[int, float, float, float, float]],
    col_off: float,
    row_off: float,
    native_tile_size: float,
    img_width: int,
    img_height: int,
    *,
    min_area_ratio: float = 0.1,
    native_res: float = 1.0,
    min_bbox_size: float | None = None,
) -> list[tuple[int, float, float, float, float]]:
    """ラベルをタイルウィンドウに合わせてクリッピング・変換する。

    Parameters
    ----------
    labels:
        ``(class_id, x_center, y_center, width, height)`` のリスト。
        座標は元画像サイズで正規化済み (0‑1)。
    col_off:
        タイルウィンドウの X オフセット (ネイティブピクセル)。
    row_off:
        タイルウィンドウの Y オフセット (ネイティブピクセル)。
    native_tile_size:
        タイルウィンドウの一辺 (ネイティブピクセル)。
    img_width:
        元画像の幅 (ピクセル)。
    img_height:
        元画像の高さ (ピクセル)。
    min_area_ratio:
        クリッピング後の面積が元面積のこの比率未満なら除外する。
    native_res:
        元画像のピクセルあたり地上分解能 (メートル)。``min_bbox_size`` を
        使う場合に bbox サイズをメートルへ変換するために用いる。
    min_bbox_size:
        クリッピング後の bbox の幅・高さの長い方がこの値 (メートル) 未満
        なら除外する。``None`` の場合は無効 (デフォルト)。

    Returns
    -------
    list[tuple[int, float, float, float, float]]
        タイル座標系で正規化 (0‑1) したラベル。
    """
    result: list[tuple[int, float, float, float, float]] = []

    tile_x2 = col_off + native_tile_size
    tile_y2 = row_off + native_tile_size

    for cls_id, xc, yc, w, h in labels:
        # Absolute pixel coords in native image
        abs_cx = xc * img_width
        abs_cy = yc * img_height
        abs_w = w * img_width
        abs_h = h * img_height

        abs_x1 = abs_cx - abs_w / 2.0
        abs_y1 = abs_cy - abs_h / 2.0
        abs_x2 = abs_cx + abs_w / 2.0
        abs_y2 = abs_cy + abs_h / 2.0

        # Intersection with tile window
        ix1 = max(abs_x1, col_off)
        iy1 = max(abs_y1, row_off)
        ix2 = min(abs_x2, tile_x2)
        iy2 = min(abs_y2, tile_y2)

        if ix1 >= ix2 or iy1 >= iy2:
            continue

        orig_area = abs_w * abs_h
        clip_area = (ix2 - ix1) * (iy2 - iy1)
        if orig_area > 0 and clip_area / orig_area < min_area_ratio:
            continue

        if min_bbox_size is not None:
            w_m = (ix2 - ix1) * native_res
            h_m = (iy2 - iy1) * native_res
            if max(w_m, h_m) < min_bbox_size:
                continue

        # Tile-relative normalised coords
        tile_cx = ((ix1 + ix2) / 2.0 - col_off) / native_tile_size
        tile_cy = ((iy1 + iy2) / 2.0 - row_off) / native_tile_size
        tile_w = (ix2 - ix1) / native_tile_size
        tile_h = (iy2 - iy1) / native_tile_size

        result.append((cls_id, tile_cx, tile_cy, tile_w, tile_h))

    return result


# ---------------------------------------------------------------------------
# rasterio I/O ラッパー
# ---------------------------------------------------------------------------

def _get_geotiff_resolution(dataset: rasterio.DatasetReader) -> float:
    """GeoTIFF のピクセルあたり地上分解能 (メートル) を返す。

    CRS が地理座標系（度単位）の場合は中心緯度を使ってメートルへ変換する。
    投影座標系（メートル単位）の場合はそのまま平均を取る。
    計算の実体は :func:`_compute_geo_resolution` に委譲する。
    """
    transform = dataset.transform
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    is_geographic = bool(dataset.crs and dataset.crs.is_geographic)
    center_lat = 0.0
    if is_geographic:
        bounds = dataset.bounds
        center_lat = (bounds.top + bounds.bottom) / 2.0

    return _compute_geo_resolution(
        res_x, res_y,
        is_geographic=is_geographic,
        center_lat=center_lat,
    )


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


def _slice_single_geotiff(
    tif_path: Path,
    labels_in: Path,
    out_images: Path,
    out_labels: Path,
    resolution: float | tuple[float, float],
    image_size: int,
    overlap: float,
    min_area_ratio: float = 0.1,
    min_bbox_size: float | None = None,
    output_geotiff: bool = False,
) -> tuple[str, int, str | None]:
    """単一 GeoTIFF を指定解像度でタイルに切り出す。

    GeoTIFF をウィンドウ単位で読み出し、各ウィンドウを ``image_size`` に
    リサンプリングして保存する。ラベルは同時にクリッピング・変換する。

    Parameters
    ----------
    min_area_ratio:
        タイル境界をまたぐ bbox のうち、クリッピング後の面積が元の面積の
        この比率以上であれば含める。0.0 で全て含める、1.0 で完全に収まる
        もののみ含める。デフォルト 0.1。
    min_bbox_size:
        クリッピング後の bbox の幅・高さの長い方がこの値 (メートル) 未満
        なら除外する。``None`` の場合は無効 (デフォルト)。
    output_geotiff:
        ``True`` の場合、タイルを GeoTIFF (CRS・transform 付き) で出力する。
        ``False`` の場合は PNG で出力する。デフォルト ``False``。

    Returns
    -------
    tuple[str, int, str | None]
        ``(stem, タイル数, エラーメッセージ)``。成功時は ``(stem, N, None)``。
    """
    stem = tif_path.stem
    label_path = labels_in / f"{stem}.txt"

    try:
        with rasterio.open(tif_path) as dataset:
            native_res = _get_geotiff_resolution(dataset)
            if native_res <= 0:
                return stem, 0, f"無効な分解能 ({native_res:.6f})"

            params = _compute_native_tile_size(
                native_res, dataset.width, dataset.height,
                image_size, resolution,
            )
            if params is None:
                return stem, 0, f"タイルサイズが画像サイズを超過"

            _target_res, native_tile_size = params

            labels = _parse_yolo_labels(label_path)

            tiles = _iter_tile_windows(
                dataset.width, dataset.height,
                native_tile_size, overlap,
            )

            tile_count = 0
            for row, col, col_off, row_off in tiles:
                window = Window(col_off, row_off,
                                native_tile_size, native_tile_size)
                data = dataset.read(
                    window=window,
                    out_shape=(dataset.count, image_size, image_size),
                    resampling=Resampling.bilinear,
                )

                tile_name = f"{stem}_{row}_{col}"

                if output_geotiff:
                    tile_transform = _compute_tile_transform(
                        dataset.transform, col_off, row_off,
                        native_tile_size, image_size,
                    )
                    profile = {
                        "driver": "GTiff",
                        "height": image_size,
                        "width": image_size,
                        "count": dataset.count,
                        "dtype": data.dtype,
                        "crs": dataset.crs,
                        "transform": tile_transform,
                    }
                    with rasterio.open(
                        out_images / f"{tile_name}.tif", "w", **profile,
                    ) as dst:
                        dst.write(data)
                else:
                    # (bands, height, width) -> (height, width, bands) for PIL
                    if data.shape[0] == 1:
                        img_array = data[0]
                    else:
                        img_array = np.transpose(data, (1, 2, 0))
                    Image.fromarray(img_array).save(
                        out_images / f"{tile_name}.png",
                    )

                tile_labels = _clip_labels_to_window(
                    labels,
                    col_off, row_off,
                    native_tile_size,
                    dataset.width, dataset.height,
                    min_area_ratio=min_area_ratio,
                    native_res=native_res,
                    min_bbox_size=min_bbox_size,
                )
                with (out_labels / f"{tile_name}.txt").open("w") as f:
                    for cls_id, xc, yc, w, h in tile_labels:
                        f.write(
                            f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n",
                        )

                tile_count += 1

        return stem, tile_count, None

    except Exception as exc:  # noqa: BLE001
        return stem, 0, str(exc)


def _slice_all_geotiffs(
    tif_files: list[Path],
    labels_in: Path,
    out_images: Path,
    out_labels: Path,
    resolution: float | tuple[float, float],
    image_size: int,
    overlap: float,
    min_area_ratio: float = 0.1,
    min_bbox_size: float | None = None,
    max_workers: int | None = None,
    max_images: int | None = None,
    output_geotiff: bool = False,
) -> tuple[int, int]:
    """GeoTIFF 群を並列処理でタイルに切り出す。

    Parameters
    ----------
    max_workers:
        プロセス数。``None`` の場合は CPU コア数を使用する。
    max_images:
        処理する最大画像数。デバッグ用。``None`` の場合は全件処理する。

    Returns
    -------
    tuple[int, int]
        ``(処理画像数, 生成タイル数)``。
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if max_images is not None:
        tif_files = tif_files[:max_images]

    stats_images = 0
    stats_tiles = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _slice_single_geotiff,
                tif_path, labels_in, out_images, out_labels,
                resolution, image_size, overlap, min_area_ratio, min_bbox_size, output_geotiff,
            ): tif_path
            for tif_path in tif_files
        }

        with tqdm(
            total=len(tif_files),
            desc="スライス",
            unit="img",
            dynamic_ncols=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                tif_path = futures[future]
                try:
                    _stem, tiles, error = future.result()
                    if error is None:
                        stats_images += 1
                        stats_tiles += tiles
                    else:
                        logger.warning("スキップ %s: %s", tif_path.name, error)
                except Exception:
                    logger.exception("画像処理中にエラー: %s", tif_path.name)
                finally:
                    pbar.update(1)

    return stats_images, stats_tiles


def slice_training_images(
    input_dir: str | Path,
    output_dir: str | Path,
    resolution: float | tuple[float, float],
    image_size: int,
    *,
    overlap: float = 0.0,
    min_area_ratio: float = 0.1,
    min_bbox_size: float | None = None,
    max_images: int | None = None,
    output_geotiff: bool = False,
) -> dict[str, int]:
    """トレーニング画像を指定分解能でスライスする。

    GeoTIFF 毎にウィンドウ読み出し + リサンプリングを行い、各タイルを
    直接 ``output_dir`` へ書き出す。

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
    min_area_ratio:
        タイル境界をまたぐ bbox のうち、クリッピング後の面積が元の面積の
        この比率以上であれば含める (0.0〜1.0)。0.0 で全て含める、1.0 で
        完全にタイル内に収まるもののみ含める。デフォルト 0.1。
    min_bbox_size:
        クリッピング後の bbox の幅・高さの長い方がこの値 (メートル) 未満
        なら除外する。``None`` の場合は無効 (デフォルト)。
    max_images:
        処理する最大画像数。デバッグ用。``None`` の場合は全件処理する。
    output_geotiff:
        ``True`` の場合、タイルを GeoTIFF (CRS・transform 付き) で出力する。
        ``False`` の場合は PNG で出力する。デフォルト ``False``。

    Returns
    -------
    dict[str, int]
        ``{"images_processed": N, "tiles_created": N}`` の統計情報。
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    images_in = input_dir / "images" / "train"
    labels_in = input_dir / "labels" / "train"
    images_out = output_dir / "images" / "train"
    labels_out = output_dir / "labels" / "train"

    tif_files = sorted(images_in.glob("*.tif"))
    if not tif_files:
        logger.warning("GeoTIFF ファイルが見つかりません: %s", images_in)
        return {"images_processed": 0, "tiles_created": 0}

    logger.info("入力画像数: %d", len(tif_files))

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    stats_images, stats_tiles = _slice_all_geotiffs(
        tif_files, labels_in, images_out, labels_out,
        resolution, image_size, overlap,
        min_area_ratio=min_area_ratio,
        min_bbox_size=min_bbox_size,
        max_images=max_images,
        output_geotiff=output_geotiff,
    )

    if stats_images == 0:
        logger.warning("処理に成功した画像がありません")

    logger.info(
        "スライス完了 — 処理画像: %d, 生成タイル: %d",
        stats_images, stats_tiles,
    )
    return {"images_processed": stats_images, "tiles_created": stats_tiles}
