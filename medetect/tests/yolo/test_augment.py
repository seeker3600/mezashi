from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from medetect.yolo.augment import RandomCloudOverlay, _split_cloud


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_png(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


@pytest.fixture()
def cloud_dir(tmp_path: Path) -> Path:
    """白い雲っぽい PNG 画像が 2 枚入ったディレクトリを返す。"""
    d = tmp_path / "clouds"
    d.mkdir()
    # RGB 画像 (白い矩形)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[16:48, 16:48] = 200
    _write_png(d / "cloud_rgb.png", rgb)
    # RGBA 画像 (アルファ付き)
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[16:48, 16:48, :3] = 220
    rgba[16:48, 16:48, 3] = 180
    _write_png(d / "cloud_rgba.png", rgba)
    return d


@pytest.fixture()
def base_img() -> np.ndarray:
    """均一な濃いグレーの 128x128 BGR 画像。"""
    return np.full((128, 128, 3), 50, dtype=np.uint8)


# ---------------------------------------------------------------------------
# _split_cloud
# ---------------------------------------------------------------------------


class TestSplitCloud:
    def test_grayscale_input_returns_bgr_and_float_alpha(self) -> None:
        """グレースケール入力は BGR 変換され、alpha が 0–1 の float32 になる。"""
        gray = np.array([[0, 128, 255]], dtype=np.uint8)
        rgb, alpha = _split_cloud(gray)
        assert rgb.shape == (1, 3, 3)
        assert alpha.dtype == np.float32
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0

    def test_rgba_input_uses_alpha_channel(self) -> None:
        """RGBA 入力はアルファチャンネルがそのまま使われる。"""
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 3] = 128
        _, alpha = _split_cloud(rgba)
        assert alpha == pytest.approx(128 / 255.0)

    def test_rgb_input_uses_luminance_as_alpha(self) -> None:
        """RGB 入力は輝度をアルファとして利用する。"""
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        _, alpha = _split_cloud(rgb)
        assert alpha == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RandomCloudOverlay.__init__
# ---------------------------------------------------------------------------


class TestRandomCloudOverlayInit:
    def test_raises_when_dir_is_empty(self, tmp_path: Path) -> None:
        """空ディレクトリを渡すと ValueError が上がる。"""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="画像が見つかりません"):
            RandomCloudOverlay(empty)

    def test_loads_cloud_paths(self, cloud_dir: Path) -> None:
        """PNG ファイルが正しく読み込まれる。"""
        t = RandomCloudOverlay(cloud_dir, p=1.0)
        assert len(t._cloud_paths) == 2  # noqa: SLF001


# ---------------------------------------------------------------------------
# RandomCloudOverlay.apply
# ---------------------------------------------------------------------------


class TestRandomCloudOverlayApply:
    def test_output_differs_from_input(self, cloud_dir: Path, base_img: np.ndarray) -> None:
        """p=1 で適用すると出力が入力と異なる（雲が重畳される）。"""
        t = RandomCloudOverlay(cloud_dir, alpha_range=(0.8, 1.0), p=1.0)
        result = t(image=base_img)["image"]
        assert not np.array_equal(result, base_img)

    def test_output_shape_matches_input(self, cloud_dir: Path, base_img: np.ndarray) -> None:
        """出力の shape は入力と同じ。"""
        t = RandomCloudOverlay(cloud_dir, p=1.0)
        result = t(image=base_img)["image"]
        assert result.shape == base_img.shape

    def test_output_dtype_is_uint8(self, cloud_dir: Path, base_img: np.ndarray) -> None:
        """出力の dtype は uint8。"""
        t = RandomCloudOverlay(cloud_dir, p=1.0)
        result = t(image=base_img)["image"]
        assert result.dtype == np.uint8

    def test_pixel_values_in_valid_range(self, cloud_dir: Path, base_img: np.ndarray) -> None:
        """ピクセル値が 0–255 に収まっている。"""
        t = RandomCloudOverlay(cloud_dir, alpha_range=(1.0, 1.0), p=1.0)
        result = t(image=base_img)["image"]
        assert result.min() >= 0
        assert result.max() <= 255

    def test_works_on_grayscale_image(self, cloud_dir: Path) -> None:
        """グレースケール入力でも shape と dtype が正しく保たれる。"""
        gray = np.full((128, 128), 50, dtype=np.uint8)
        t = RandomCloudOverlay(cloud_dir, p=1.0)
        result = t(image=gray)["image"]
        assert result.shape == gray.shape
        assert result.dtype == np.uint8

    def test_no_change_when_p0(self, cloud_dir: Path, base_img: np.ndarray) -> None:
        """p=0 では画像が変化しない。"""
        t = RandomCloudOverlay(cloud_dir, p=0.0)
        result = t(image=base_img)["image"]
        np.testing.assert_array_equal(result, base_img)

    def test_full_coverage_with_extreme_translation(
        self, cloud_dir: Path, base_img: np.ndarray
    ) -> None:
        """rel_tx/rel_ty が極端な値でも出力が入力と同じ shape になる（全体カバー保証）。"""
        t = RandomCloudOverlay(cloud_dir, alpha_range=(0.5, 0.5), p=1.0)
        for rel_tx, rel_ty in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            params = t.get_params()
            params.update({"rel_tx": rel_tx, "rel_ty": rel_ty})
            result = t.apply(base_img, **params)
            assert result.shape == base_img.shape

    def test_rotation_produces_different_results(
        self, cloud_dir: Path, base_img: np.ndarray
    ) -> None:
        """異なる回転角では（高い確率で）異なる結果が得られる。"""
        t = RandomCloudOverlay(cloud_dir, alpha_range=(1.0, 1.0), rotation_range=(0.0, 360.0), p=1.0)
        params_a = t.get_params()
        params_a.update({"rotation": 0.0, "rel_tx": 0.5, "rel_ty": 0.5})
        params_b = t.get_params()
        params_b.update({"rotation": 45.0, "rel_tx": 0.5, "rel_ty": 0.5})
        result_a = t.apply(base_img, **params_a)
        result_b = t.apply(base_img, **params_b)
        # 45° 回転で少なくとも一部のピクセルは変化するはず
        assert not np.array_equal(result_a, result_b)
