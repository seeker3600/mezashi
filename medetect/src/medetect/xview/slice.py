"""トレーニング画像を指定分解能でスライスするモジュール。

処理概要:
  - 入力: GeoTIFF 画像 (入力フォルダ/images/train) + YOLO ラベル (入力フォルダ/labels/train)
  - 各画像を rasterio で指定の分解能にリサンプリングし、yolo-tiling でタイルに切り出す
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
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from tqdm import tqdm
from yolo_tiler import TileConfig, TileProgress, YoloTiler

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


def _compute_resample_params(
    native_res: float,
    src_width: int,
    src_height: int,
    image_size: int,
    resolution: float | tuple[float, float],
) -> tuple[float, int, int] | None:
    """リサンプリング後の ``(target_res, new_width, new_height)`` を計算する。

    スライス後のサイズが ``image_size`` 未満になる場合は ``None`` を返す。

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
    scale_factor = native_res / target_res

    new_width = max(1, int(src_width * scale_factor))
    new_height = max(1, int(src_height * scale_factor))

    if new_width < image_size or new_height < image_size:
        return None

    return target_res, new_width, new_height


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


def _resample_single(
    tif_path: Path,
    labels_in: Path,
    tmp_images: Path,
    tmp_labels: Path,
    resolution: float | tuple[float, float],
    image_size: int,
) -> tuple[str, str | None]:
    """単一 GeoTIFF をリサンプリングして一時ディレクトリへ保存する。

    Returns
    -------
    tuple[str, str | None]
        ``(stem, エラーメッセージ)``。成功時は ``(stem, None)``。
    """
    stem = tif_path.stem
    label_path = labels_in / f"{stem}.txt"

    try:
        with rasterio.open(tif_path) as dataset:
            native_res = _get_geotiff_resolution(dataset)
            if native_res <= 0:
                return stem, f"無効な分解能 ({native_res:.6f})"

            params = _compute_resample_params(
                native_res, dataset.width, dataset.height,
                image_size, resolution,
            )
            if params is None:
                return stem, f"リサンプリング後サイズが image_size ({image_size}) 未満"

            _target_res, new_width, new_height = params

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

        Image.fromarray(img_array).save(tmp_images / f"{stem}.png")
        del img_array

        if label_path.exists():
            shutil.copy2(label_path, tmp_labels / f"{stem}.txt")
        else:
            (tmp_labels / f"{stem}.txt").touch()

        return stem, None

    except Exception as exc:  # noqa: BLE001
        return stem, str(exc)


def _resample_to_tmpdir(
    tif_files: list[Path],
    labels_in: Path,
    tmp_images: Path,
    tmp_labels: Path,
    resolution: float | tuple[float, float],
    image_size: int,
    max_workers: int | None = None,
) -> int:
    """GeoTIFF 群を並列リサンプリングし一時ディレクトリへ保存する。

    Parameters
    ----------
    max_workers:
        プロセス数。``None`` の場合は CPU コア数を使用する。

    Returns
    -------
    int
        処理に成功した画像数。
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    
    # デバッグ目的で処理数を制限する場合はここでスライス
    # tif_files = tif_files[:max_workers]

    stats_images = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _resample_single,
                tif_path, labels_in, tmp_images, tmp_labels,
                resolution, image_size,
            ): tif_path
            for tif_path in tif_files
        }

        with tqdm(
            total=len(tif_files),
            desc="リサンプリング",
            unit="img",
            dynamic_ncols=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                tif_path = futures[future]
                try:
                    _stem, error = future.result()
                    if error is None:
                        stats_images += 1
                    else:
                        logger.warning("スキップ %s: %s", tif_path.name, error)
                except Exception:
                    logger.exception("画像処理中にエラー: %s", tif_path.name)
                finally:
                    pbar.update(1)

    return stats_images


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

    tif_files = sorted(images_in.glob("*.tif"))
    if not tif_files:
        logger.warning("GeoTIFF ファイルが見つかりません: %s", images_in)
        return {"images_processed": 0}

    logger.info("入力画像数: %d", len(tif_files))

    # ── 1. rasterio でリサンプリング → 一時ディレクトリへ保存 ──
    # output_dir と同じドライブに一時ディレクトリを作成することで move が rename になり高速化
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as tmp_base_str:
        tmp_base = Path(tmp_base_str)

        # yolo-tiling が期待する構造: source/train/images/, source/train/labels/
        # valid_ratio=0.0 でも valid フォルダが必要
        tmp_source = tmp_base / "source"
        tmp_images = tmp_source / "train" / "images"
        tmp_labels = tmp_source / "train" / "labels"
        tmp_images.mkdir(parents=True, exist_ok=True)
        tmp_labels.mkdir(parents=True, exist_ok=True)
        (tmp_source / "valid" / "images").mkdir(parents=True, exist_ok=True)
        (tmp_source / "valid" / "labels").mkdir(parents=True, exist_ok=True)
        (tmp_source / "test" / "images").mkdir(parents=True, exist_ok=True)
        (tmp_source / "test" / "labels").mkdir(parents=True, exist_ok=True)

        stats_images = _resample_to_tmpdir(
            tif_files, labels_in, tmp_images, tmp_labels,
            resolution, image_size,
        )

        if stats_images == 0:
            logger.warning("リサンプリングに成功した画像がありません")
            return {"images_processed": 0}

        # ── 2. yolo-tiling でスライス ──
        tmp_target = tmp_base / "target"

        config = TileConfig(
            slice_wh=(image_size, image_size),
            overlap_wh=(overlap, overlap),
            annotation_type="object_detection",
            output_ext=".png",
            train_ratio=1.0,
            valid_ratio=0.0,
            test_ratio=0.0,
            include_negative_samples=True,
        )

        tiler = YoloTiler(
            source=str(tmp_source),
            target=str(tmp_target),
            config=config,
            num_viz_samples=0,
            show_processing_status=False,
        )

        # YoloTiler の INFO ログを抑止し、2本の tqdm バーで進捗表示
        _yolo_logger = logging.getLogger("YoloTiler")
        _yolo_logger.setLevel(logging.WARNING)

        with tqdm(unit="img", unit_scale=False, dynamic_ncols=True) as pbar, \
            tqdm(unit="tile", unit_scale=False, dynamic_ncols=True) as tile_pbar:

            def _progress_callback(p: TileProgress) -> None:
                nonlocal pbar, tile_pbar
                pbar.desc = p.current_set_name
                pbar.total = p.total_images
                pbar.n = p.current_image_idx
                pbar.refresh()

                tile_pbar.desc = p.current_image_name
                tile_pbar.total = p.total_tiles
                tile_pbar.n = p.current_tile_idx
                tile_pbar.refresh()

            tiler.progress_callback = _progress_callback
            tiler.run()

        logger.info("yolo-tiling によるスライス完了")

        # ── 3. タイル出力ディレクトリを最終ディレクトリへ移動 ──
        shutil.move(str(tmp_target / "train" / "images"), str(images_out))
        shutil.move(str(tmp_target / "train" / "labels"), str(labels_out))

    logger.info("スライス完了 — リサンプリング画像: %d", stats_images)
    return {"images_processed": stats_images}
