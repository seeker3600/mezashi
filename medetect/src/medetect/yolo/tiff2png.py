"""TIFF / GeoTIFF ファイルを PNG へ一括変換するモジュール。

処理概要:
  - 指定ディレクトリ以下を再帰して .tif / .tiff / .geotiff を検出する。
  - rasterio で読み込み、uint8 RGB / グレースケール PNG として書き出す。
  - 変換後に元の TIFF ファイルを削除する（``delete_source=False`` で無効化可能）。
  - ThreadPoolExecutor による並列処理と tqdm による進捗表示を提供する。

使い方::

    from medetect.yolo.tiff2png import convert_tiffs_to_png

    success, errors = convert_tiffs_to_png("datasets/xView/images/train")
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

_TIFF_EXTENSIONS = frozenset({".tif", ".tiff", ".geotiff"})


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """NumPy 配列を uint8 へ変換する。

    - ``uint8``: そのまま返す。
    - 浮動小数点: min-max 正規化して 0–255 にスケールする。
    - 整数型: データ型の全範囲を 0–255 にスケールする。
    """
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin:
            scaled = (arr - vmin) / (vmax - vmin) * 255.0
        else:
            scaled = np.zeros_like(arr, dtype=np.float32)
        return scaled.astype(np.uint8)
    # 整数型: データ型の全範囲でスケール
    info = np.iinfo(arr.dtype)
    scale = 255.0 / max(info.max - info.min, 1)
    return ((arr.astype(np.float64) - info.min) * scale).astype(np.uint8)


def _convert_single_tiff(src_path: Path, *, delete_source: bool) -> str | None:
    """単一 TIFF ファイルを同名 PNG へ変換する。

    Returns
    -------
    str | None
        エラーメッセージ。成功時は ``None``。
    """
    dst_path = src_path.with_suffix(".png")
    try:
        with rasterio.open(src_path) as ds:
            data = ds.read()  # shape: (bands, height, width)
            count = data.shape[0]

            if count == 1:
                img = Image.fromarray(_to_uint8(data[0]), mode="L")
            elif count >= 3:
                rgb = np.stack([_to_uint8(data[i]) for i in range(3)], axis=-1)
                img = Image.fromarray(rgb, mode="RGB")
            else:  # 2 バンド: 最初のバンドをグレースケールとして使用
                img = Image.fromarray(_to_uint8(data[0]), mode="L")

        img.save(dst_path, format="PNG")

    except Exception as exc:  # noqa: BLE001
        return str(exc)

    if delete_source:
        src_path.unlink()

    return None


def convert_tiffs_to_png(
    directory: str | Path,
    *,
    delete_source: bool = True,
    max_workers: int | None = None,
) -> tuple[int, int]:
    """指定ディレクトリ以下の TIFF ファイルを再帰的に PNG へ変換する。

    Parameters
    ----------
    directory:
        検索対象のルートディレクトリ。
    delete_source:
        変換後に元の TIFF ファイルを削除するかどうか。デフォルト ``True``。
    max_workers:
        スレッド数。``None`` の場合は CPU コア数を使用する。

    Returns
    -------
    tuple[int, int]
        ``(変換成功数, エラー数)``。
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {root}")

    tiff_files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in _TIFF_EXTENSIONS
    )

    if not tiff_files:
        logger.info("TIFF ファイルが見つかりませんでした: %s", root)
        return 0, 0

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    success = 0
    errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_convert_single_tiff, p, delete_source=delete_source): p
            for p in tiff_files
        }

        with tqdm(
            total=len(tiff_files),
            desc="tiff2png",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                tiff_path = futures[future]
                try:
                    error = future.result()
                    if error is None:
                        success += 1
                    else:
                        errors += 1
                        logger.warning("変換失敗 %s: %s", tiff_path.name, error)
                except Exception:
                    errors += 1
                    logger.exception("変換中にエラー: %s", tiff_path.name)
                finally:
                    pbar.update(1)

    logger.info("完了: 成功 %d 件, エラー %d 件", success, errors)
    return success, errors
