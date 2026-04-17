from __future__ import annotations

import pathlib
import random

import numpy as np
import pytest

from medetect.datagen.compose import _compose_one, augment_tile, is_dark_tile, make_nodata_mask


class TestGeoScale:
    """geo_scale モードのテスト。"""

    @pytest.fixture()
    def tiny_tif(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """低解像度(100 m/px)の小さな GeoTIFF を生成する。"""
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "bg.tif"
        size = 100
        transform = from_bounds(0, 0, 100 * size, 100 * size, size, size)
        data = np.full((3, size, size), 128, dtype=np.uint8)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(32654),
            transform=transform,
        ) as dst:
            dst.write(data)
        return tif_path

    def test_geo_scale_1_produces_correct_size(self, tiny_tif: pathlib.Path) -> None:
        """geo_scale=1.0 のとき出力タイルが image_size×image_size になる。"""
        rng = random.Random(0)
        result = _compose_one(
            tif_path=tiny_tif,
            svg_metas=None,
            image_size=64,
            resolution=10.0,
            geo_scale=1.0,
            ships_per_image=(0, 0),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            ship_blur_sigma=0.5,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            rng=rng,
        )
        assert result is not None
        tile, _labels, _n_clusters = result
        assert tile.shape == (64, 64, 3)

    def test_geo_scale_none_uses_crs(self, tiny_tif: pathlib.Path) -> None:
        """geo_scale=None のときは CRS ベースの解像度変換が行われる。"""
        rng = random.Random(1)
        result = _compose_one(
            tif_path=tiny_tif,
            svg_metas=None,
            image_size=64,
            resolution=10.0,
            geo_scale=None,
            ships_per_image=(0, 0),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            ship_blur_sigma=0.5,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            rng=rng,
        )
        assert result is not None
        tile, _, _ = result
        assert tile.shape == (64, 64, 3)

    def test_geo_scale_05_upsamples(self, tiny_tif: pathlib.Path) -> None:
        """geo_scale=0.5 のとき半分の TIFF ピクセルを読みアップサンプルする。"""
        rng = random.Random(2)
        result = _compose_one(
            tif_path=tiny_tif,
            svg_metas=None,
            image_size=64,
            resolution=10.0,
            geo_scale=0.5,
            ships_per_image=(0, 0),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            ship_blur_sigma=0.5,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            rng=rng,
        )
        assert result is not None
        tile, _, _ = result
        assert tile.shape == (64, 64, 3)


class TestLandOnlyNegativeExample:
    """陸地のみのタイルがネガティブサンプルとして返るテスト。"""

    @pytest.fixture()
    def land_only_tif(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """水域なし（全面陸地相当）の GeoTIFF を生成する。"""
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "land_visual.tif"
        size = 200
        transform = from_bounds(0, 0, 100 * size, 100 * size, size, size)
        data = np.full((3, size, size), 0, dtype=np.uint8)
        data[0, :, :] = 120
        data[1, :, :] = 100
        data[2, :, :] = 70
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=3,
            dtype="uint8",
            crs=CRS.from_epsg(32654),
            transform=transform,
        ) as dst:
            dst.write(data)
        return tif_path

    def test_land_tile_returns_negative_example(self, land_only_tif: pathlib.Path) -> None:
        """水域なしタイルが船なし（ラベル空）で返る。"""
        rng = random.Random(42)
        result = _compose_one(
            tif_path=land_only_tif,
            svg_metas=None,
            image_size=64,
            resolution=10.0,
            geo_scale=1.0,
            ships_per_image=(3, 5),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.3,
            ship_blur_sigma=0.5,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            rng=rng,
        )
        assert result is not None
        tile, labels, n_clusters = result
        assert tile.shape == (64, 64, 3)
        assert labels == []
        assert n_clusters == 0


class TestIsDarkTile:
    """衛星画像の帯状真っ黒領域の検出テスト。"""

    def test_all_black_is_dark(self) -> None:
        """全黒タイルは暗いと判定される。"""
        tile = np.zeros((64, 64, 3), dtype=np.uint8)
        assert is_dark_tile(tile)

    def test_normal_image_is_not_dark(self) -> None:
        """通常の輝度を持つタイルは暗いと判定されない。"""
        tile = np.full((64, 64, 3), 80, dtype=np.uint8)
        assert not is_dark_tile(tile)

    def test_custom_threshold(self) -> None:
        """カスタム閾値が適用される。"""
        tile = np.full((64, 64, 3), 5, dtype=np.uint8)
        assert not is_dark_tile(tile, threshold=5.0)
        assert is_dark_tile(tile, threshold=6.0)

    def test_stripe_scenario(self) -> None:
        """大半が黒いストライプ状タイルは暗いと判定される。"""
        tile = np.full((64, 64, 3), 80, dtype=np.uint8)
        tile[:32, :, :] = 0
        assert not is_dark_tile(tile)

        mostly_dark = np.zeros((64, 64, 3), dtype=np.uint8)
        mostly_dark[60:64, 60:64, :] = 80
        assert is_dark_tile(mostly_dark)


class TestMakeNodataMask:
    """純黒 (#000000) の no-data 領域検出テスト。"""

    def test_all_black_is_nodata(self) -> None:
        """全黒タイルは全ピクセルが no-data とマークされる。"""
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = make_nodata_mask(tile)
        assert mask.all()

    def test_normal_tile_has_no_nodata(self) -> None:
        """通常の輝度を持つタイルは no-data ピクセルを含まない。"""
        tile = np.full((4, 4, 3), 80, dtype=np.uint8)
        mask = make_nodata_mask(tile)
        assert not mask.any()

    def test_partial_black_strip(self) -> None:
        """一部が黒いストライプを含むタイルは黒部分のみ no-data。"""
        tile = np.full((8, 8, 3), 100, dtype=np.uint8)
        tile[:4, :, :] = 0
        mask = make_nodata_mask(tile)
        assert mask[:4, :].all()
        assert not mask[4:, :].any()

    def test_single_channel_zero_is_not_nodata(self) -> None:
        """1チャンネルだけ 0 でも no-data ではない。"""
        tile = np.full((4, 4, 3), 0, dtype=np.uint8)
        tile[:, :, 0] = 50
        mask = make_nodata_mask(tile)
        assert not mask.any()

    def test_nodata_excluded_from_water_mask(self) -> None:
        """pure black ピクセルは RGB ウォーターマスクから除外される。"""
        from medetect.datagen.water_mask import make_water_mask_from_rgb

        tile = np.zeros((8, 8, 3), dtype=np.uint8)
        tile[4:, :, :] = 30

        water = make_water_mask_from_rgb(tile)
        nodata = make_nodata_mask(tile)
        water_clean = water & ~nodata

        assert not water_clean[:4, :].any()
        assert water_clean[4:, :].any()


class TestAugmentTile:
    """augment_tile によるタイルの色オーグメンテーション検証。"""

    def test_output_shape_unchanged(self) -> None:
        """出力タイルの形状が変わらない。"""
        tile = np.full((64, 64, 3), 100, dtype=np.uint8)
        rng = random.Random(0)
        result = augment_tile(tile, rng)
        assert result.shape == tile.shape
        assert result.dtype == np.uint8

    def test_output_differs_from_input(self) -> None:
        """オーグメンテーション後は元と異なる値になる。"""
        tile = np.full((64, 64, 3), 100, dtype=np.uint8)
        rng = random.Random(42)
        result = augment_tile(tile, rng)
        assert not np.array_equal(result, tile)

    def test_different_seeds_produce_different_results(self) -> None:
        """異なるシードで異なるオーグメンテーション結果になる。"""
        tile = np.full((64, 64, 3), 80, dtype=np.uint8)
        results = []
        for seed in range(10):
            rng = random.Random(seed)
            results.append(augment_tile(tile, rng).mean())
        unique_means = set(round(mean, 1) for mean in results)
        assert len(unique_means) >= 3

    def test_values_clipped_to_uint8(self) -> None:
        """オーグメンテーション後の値が 0-255 に収まる。"""
        for value in [0, 5, 250, 255]:
            tile = np.full((32, 32, 3), value, dtype=np.uint8)
            for seed in range(5):
                rng = random.Random(seed)
                result = augment_tile(tile, rng)
                assert result.min() >= 0
                assert result.max() <= 255

    def test_channels_shifted_independently(self) -> None:
        """チャンネル別のゲインが独立に適用される。"""
        tile = np.full((64, 64, 3), 100, dtype=np.uint8)
        channel_diffs = 0
        for seed in range(20):
            rng = random.Random(seed)
            result = augment_tile(tile, rng)
            means = [result[:, :, channel].mean() for channel in range(3)]
            if max(means) - min(means) > 1:
                channel_diffs += 1
        assert channel_diffs > 0

    def test_typical_water_tile_augmented(self) -> None:
        """典型的な暗い海面タイルに色の多様性が出る。"""
        tile = np.zeros((64, 64, 3), dtype=np.uint8)
        tile[:, :, 0] = 15
        tile[:, :, 1] = 25
        tile[:, :, 2] = 40
        means = []
        for seed in range(20):
            rng = random.Random(seed)
            result = augment_tile(tile, rng)
            means.append(tuple(result.mean(axis=(0, 1))))
        blue_means = [mean[2] for mean in means]
        assert max(blue_means) - min(blue_means) > 5