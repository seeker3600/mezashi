from __future__ import annotations

import pathlib
import random
import math

import numpy as np
import pytest

import medetect.datagen.compose as compose_mod

from medetect.datagen.compose import _compose_one, augment_tile, is_dark_tile, make_nodata_mask


@pytest.fixture()
def water_tif(tmp_path: pathlib.Path) -> pathlib.Path:
    """全面水域として扱える簡単な GeoTIFF を生成する。"""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    tif_path = tmp_path / "water_visual.tif"
    size = 128
    transform = from_bounds(0, 0, 10 * size, 10 * size, size, size)
    data = np.zeros((3, size, size), dtype=np.uint8)
    data[0, :, :] = 20
    data[1, :, :] = 60
    data[2, :, :] = 90
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
            edge_hardness=0.75,
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
            edge_hardness=0.75,
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
            edge_hardness=0.75,
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
            edge_hardness=0.75,
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

    def test_debug_bg_color_skips_augmentation(
        self,
        water_tif: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """debug_bg_color 指定時は背景が単色化され augment_tile を通らない。"""

        def _fail_augment(tile: np.ndarray, rng: random.Random) -> np.ndarray:
            del tile, rng
            msg = "augment_tile should not run when debug_bg_color is set"
            raise AssertionError(msg)

        monkeypatch.setattr(compose_mod, "augment_tile", _fail_augment)

        result = _compose_one(
            tif_path=water_tif,
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
            edge_hardness=0.75,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            rng=random.Random(0),
            debug_bg_color=(0x12, 0x34, 0x56),
        )

        assert result is not None
        tile, labels, n_clusters = result
        assert labels == []
        assert n_clusters == 0
        assert np.all(tile[:, :, 0] == 0x12)
        assert np.all(tile[:, :, 1] == 0x34)
        assert np.all(tile[:, :, 2] == 0x56)


class TestComposeShadows:
    """_compose_one における影レイヤ順と方向のテスト。"""

    def test_single_ship_shadows_render_between_wakes_and_hulls(
        self,
        water_tif: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """単船の影は wake の後、船体の前に描かれる。"""
        call_order: list[str] = []
        shadow_lengths: list[float] = []
        positions = [(24, 24), (48, 48)]
        ship_rgba = np.zeros((12, 6, 4), dtype=np.uint8)
        ship_rgba[:, :, :3] = 180
        ship_rgba[:, :, 3] = 255

        monkeypatch.setattr(compose_mod, "make_water_mask_from_rgb", lambda tile: np.ones(tile.shape[:2], dtype=bool))
        monkeypatch.setattr(compose_mod, "augment_tile", lambda tile, rng: tile)
        monkeypatch.setattr(compose_mod, "_pick_svg", lambda *args, **kwargs: "<svg/>")
        monkeypatch.setattr(
            compose_mod,
            "_render_ship",
            lambda *args, **kwargs: (ship_rgba.copy(), "mock", 6, 12, 2.0),
        )
        monkeypatch.setattr(
            compose_mod,
            "find_water_position",
            lambda *args, **kwargs: positions.pop(0),
        )
        monkeypatch.setattr(
            compose_mod,
            "_sample_water_tint",
            lambda *args, **kwargs: np.array([40.0, 50.0, 60.0], dtype=np.float32),
        )
        monkeypatch.setattr(compose_mod, "render_wake", lambda *args, **kwargs: call_order.append("wake"))
        monkeypatch.setattr(
            compose_mod,
            "_shadow_offset_pixels",
            lambda beam_px, length_px, azimuth_rad, shadow_length, *, scene_scale=1: shadow_lengths.append(shadow_length) or (3, 1),
        )
        monkeypatch.setattr(compose_mod, "_shadow_blur_sigma", lambda *args, **kwargs: 1.0)
        monkeypatch.setattr(compose_mod, "_shadow_alpha_for_ship", lambda *args, **kwargs: 0.4)

        def _mock_make_shadow_rgba(
            ship_rgba: np.ndarray,
            *,
            offset_x: int,
            offset_y: int,
            blur_sigma: float,
            alpha_scale: float,
        ) -> np.ndarray:
            return np.zeros((ship_rgba.shape[0] + 4, ship_rgba.shape[1] + 4, 4), dtype=np.uint8)

        monkeypatch.setattr(compose_mod, "_make_shadow_rgba", _mock_make_shadow_rgba)
        monkeypatch.setattr(compose_mod, "blend_shadow", lambda *args, **kwargs: call_order.append("shadow"))
        monkeypatch.setattr(compose_mod, "blend_ship", lambda *args, **kwargs: call_order.append("ship"))

        result = _compose_one(
            tif_path=water_tif,
            svg_metas=None,
            image_size=64,
            resolution=10.0,
            geo_scale=1.0,
            ships_per_image=(2, 2),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            edge_hardness=1.0,
            ship_alpha=(1.0, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            shadow_alpha_scale=1.0,
            shadow_length_range=(2.5, 2.5),
            rng=random.Random(7),
        )

        assert result is not None
        assert call_order == ["wake", "wake", "shadow", "shadow", "ship", "ship"]
        assert len(shadow_lengths) == 2
        assert len({round(value, 6) for value in shadow_lengths}) == 1
        assert shadow_lengths[0] == pytest.approx(2.5)

    def test_single_ship_shadows_share_tile_alpha_with_size_bias(
        self,
        water_tif: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """単船の影は画像単位の基準濃さを共有しつつ大型船だけ少し濃い。"""
        positions = [(24, 24), (48, 48)]
        ship_rgba = np.zeros((12, 6, 4), dtype=np.uint8)
        ship_rgba[:, :, :3] = 180
        ship_rgba[:, :, 3] = 255
        ship_specs = [
            (ship_rgba.copy(), "small", 6, 12, 2.0),
            (ship_rgba.copy(), "large", 12, 24, 2.0),
        ]
        shadow_patch_biases: list[float] = []
        shadow_blend_factors: list[float] = []

        monkeypatch.setattr(compose_mod, "make_water_mask_from_rgb", lambda tile: np.ones(tile.shape[:2], dtype=bool))
        monkeypatch.setattr(compose_mod, "augment_tile", lambda tile, rng: tile)
        monkeypatch.setattr(compose_mod, "_pick_svg", lambda *args, **kwargs: "<svg/>")
        monkeypatch.setattr(compose_mod, "_render_ship", lambda *args, **kwargs: ship_specs.pop(0))
        monkeypatch.setattr(compose_mod, "find_water_position", lambda *args, **kwargs: positions.pop(0))
        monkeypatch.setattr(
            compose_mod,
            "_sample_water_tint",
            lambda *args, **kwargs: np.array([40.0, 50.0, 60.0], dtype=np.float32),
        )
        monkeypatch.setattr(compose_mod, "render_wake", lambda *args, **kwargs: None)
        monkeypatch.setattr(compose_mod, "_sample_shadow_alpha", lambda rng: 0.09)
        monkeypatch.setattr(compose_mod, "_shadow_offset_pixels", lambda *args, **kwargs: (3, 1))
        monkeypatch.setattr(compose_mod, "_shadow_blur_sigma", lambda *args, **kwargs: 1.0)
        monkeypatch.setattr(
            compose_mod,
            "_shadow_alpha_for_ship",
            lambda beam_px, length_px: 1.0 if beam_px == 6 else 1.08,
        )

        def _mock_make_shadow_rgba(
            ship_rgba: np.ndarray,
            *,
            offset_x: int,
            offset_y: int,
            blur_sigma: float,
            alpha_scale: float,
        ) -> np.ndarray:
            del offset_x, offset_y, blur_sigma
            shadow_patch_biases.append(alpha_scale)
            return np.zeros((ship_rgba.shape[0] + 4, ship_rgba.shape[1] + 4, 4), dtype=np.uint8)

        def _mock_blend_shadow(
            background: np.ndarray,
            shadow_rgba: np.ndarray,
            cx: int,
            cy: int,
            alpha_factor: float = 1.0,
            clip_mask: np.ndarray | None = None,
        ) -> None:
            del background, shadow_rgba, cx, cy, clip_mask
            shadow_blend_factors.append(alpha_factor)

        monkeypatch.setattr(compose_mod, "_make_shadow_rgba", _mock_make_shadow_rgba)
        monkeypatch.setattr(compose_mod, "blend_shadow", _mock_blend_shadow)
        monkeypatch.setattr(compose_mod, "blend_ship", lambda *args, **kwargs: None)

        result = _compose_one(
            tif_path=water_tif,
            svg_metas=None,
            image_size=64,
            resolution=10.0,
            geo_scale=1.0,
            ships_per_image=(2, 2),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            edge_hardness=1.0,
            ship_alpha=(1.0, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            shadow_alpha_scale=2.0,
            shadow_length_range=(2.5, 2.5),
            rng=random.Random(9),
        )

        assert result is not None
        assert shadow_patch_biases == [1.0, 1.08]
        assert shadow_blend_factors == [pytest.approx(0.18), pytest.approx(0.18)]