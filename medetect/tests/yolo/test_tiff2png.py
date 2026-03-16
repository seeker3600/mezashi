"""medetect.yolo.tiff2png のテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from medetect.yolo.tiff2png import _to_uint8, convert_tiffs_to_png


def _write_tiff(path: Path, data: np.ndarray) -> None:
    """テスト用の単バンドまたは多バンド TIFF を書き出す。

    data shape: (bands, height, width) または (height, width) for 1-band。
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # (1, H, W)
    bands, height, width = data.shape
    transform = from_bounds(0, 0, 1, 1, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as ds:
        ds.write(data)


class TestToUint8:
    def test_uint8_passthrough(self) -> None:
        """uint8 配列はそのまま返す。"""
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = _to_uint8(arr)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, arr)

    def test_uint16_scaled(self) -> None:
        """uint16 はデータ型全範囲でスケールされ 0–255 になる。"""
        arr = np.array([0, 32768, 65535], dtype=np.uint16)
        result = _to_uint8(arr)
        assert result.dtype == np.uint8
        assert int(result[0]) == 0
        assert int(result[2]) == 255

    def test_float32_normalized(self) -> None:
        """float32 は min-max 正規化される。"""
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = _to_uint8(arr)
        assert result.dtype == np.uint8
        assert int(result[0]) == 0
        assert int(result[2]) == 255

    def test_float_constant_array(self) -> None:
        """全値が同一の float 配列はゼロ埋めになる。"""
        arr = np.full((3,), 0.5, dtype=np.float32)
        result = _to_uint8(arr)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.uint8))


class TestConvertTiffsToPng:
    def test_grayscale_tiff_converted(self, tmp_path: Path) -> None:
        """単バンド TIFF が PNG に変換され、元 TIFF が削除される。"""
        src = tmp_path / "gray.tif"
        arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
        _write_tiff(src, arr)

        success, errors = convert_tiffs_to_png(tmp_path, delete_source=True, max_workers=1)

        assert success == 1
        assert errors == 0
        assert not src.exists()
        assert (tmp_path / "gray.png").exists()

    def test_rgb_tiff_converted(self, tmp_path: Path) -> None:
        """3 バンド TIFF が PNG に変換される。"""
        src = tmp_path / "rgb.tiff"
        arr = np.ones((3, 4, 4), dtype=np.uint8) * 127
        _write_tiff(src, arr)

        success, errors = convert_tiffs_to_png(tmp_path, delete_source=True, max_workers=1)

        assert success == 1
        assert errors == 0
        assert (tmp_path / "rgb.png").exists()

    def test_keep_source(self, tmp_path: Path) -> None:
        """``delete_source=False`` のとき元 TIFF が残る。"""
        src = tmp_path / "keep.tif"
        arr = np.zeros((4, 4), dtype=np.uint8)
        _write_tiff(src, arr)

        convert_tiffs_to_png(tmp_path, delete_source=False, max_workers=1)

        assert src.exists()
        assert (tmp_path / "keep.png").exists()

    def test_recursive_search(self, tmp_path: Path) -> None:
        """サブディレクトリ内の TIFF も再帰的に変換される。"""
        sub = tmp_path / "sub" / "nested"
        sub.mkdir(parents=True)
        src = sub / "nested.tif"
        arr = np.zeros((4, 4), dtype=np.uint8)
        _write_tiff(src, arr)

        success, errors = convert_tiffs_to_png(tmp_path, delete_source=True, max_workers=1)

        assert success == 1
        assert errors == 0
        assert (sub / "nested.png").exists()

    def test_no_tiff_files(self, tmp_path: Path) -> None:
        """TIFF が存在しないとき (0, 0) を返す。"""
        success, errors = convert_tiffs_to_png(tmp_path, max_workers=1)
        assert success == 0
        assert errors == 0

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        """存在しないディレクトリを渡すと FileNotFoundError になる。"""
        with pytest.raises(FileNotFoundError):
            convert_tiffs_to_png(tmp_path / "does_not_exist")

    def test_geotiff_extension(self, tmp_path: Path) -> None:
        """.geotiff 拡張子のファイルも変換対象になる。"""
        src = tmp_path / "sample.geotiff"
        arr = np.zeros((4, 4), dtype=np.uint8)
        _write_tiff(src, arr)

        success, errors = convert_tiffs_to_png(tmp_path, delete_source=True, max_workers=1)

        assert success == 1
        assert (tmp_path / "sample.png").exists()

    def test_uint16_converted_correctly(self, tmp_path: Path) -> None:
        """uint16 TIFF が uint8 PNG として保存される。"""
        src = tmp_path / "u16.tif"
        arr = np.array([[0, 32768], [65535, 16384]], dtype=np.uint16)
        _write_tiff(src, arr)

        convert_tiffs_to_png(tmp_path, delete_source=False, max_workers=1)

        from PIL import Image
        img = Image.open(tmp_path / "u16.png")
        assert img.mode == "L"
        pixels = np.array(img)
        assert pixels.dtype == np.uint8
