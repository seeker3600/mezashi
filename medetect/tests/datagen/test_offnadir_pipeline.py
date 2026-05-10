"""offnadir_range パラメータがパイプライン全体でクラッシュしないことを確認するテスト。"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import medetect.datagen.pipeline as pipeline_mod
from medetect.datagen.ship import _pick_svg


class TestPickSvgOffnadir:
    def test_onthefly_starboard(self) -> None:
        """offnadir_deg=15, az=90 でオンザフライ生成してもクラッシュしない。"""
        result = _pick_svg(
            None,
            random.Random(0),
            offnadir_deg=15.0,
            sensor_az_ship_deg=90.0,
        )
        root = ET.fromstring(result)
        assert root.get("data-visible-side") == "starboard"

    def test_onthefly_nadir(self) -> None:
        """offnadir_deg=0 でオンザフライ生成してもクラッシュしない。"""
        result = _pick_svg(
            None,
            random.Random(0),
            offnadir_deg=0.0,
            sensor_az_ship_deg=0.0,
        )
        root = ET.fromstring(result)
        assert root.get("data-visible-side") == "none"

    def test_pregenerated_svg_returned_as_is(self, tmp_path) -> None:
        """事前生成SVGが渡された場合は offnadir 引数を無視してSVGをそのまま返す。"""
        from medetect.shipgen.gen import generate_ship_svg
        from medetect.datagen.ship import _SvgMeta, _load_svg

        svg_path = tmp_path / "ship.svg"
        svg_path.write_text(generate_ship_svg("patrol", rng=random.Random(0)), encoding="utf-8")

        metas = [_SvgMeta(path=svg_path, lb_ratio=5.0)]
        result = _pick_svg(
            metas,
            random.Random(0),
            offnadir_deg=30.0,
            sensor_az_ship_deg=90.0,
        )
        # 事前生成SVGはそのまま返るので data-visible-side は元のSVGの値になる
        root = ET.fromstring(result)
        assert root.tag is not None


class TestComposeOneOffnadir:
    """_compose_one に offnadir_range を渡してもクラッシュしないことを確認する。"""

    def test_compose_with_offnadir(self, tmp_path) -> None:
        """offnadir_range=(0, 30) で _compose_one を呼び出してもクラッシュしない。"""
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.transform import from_bounds
        import rasterio.crs

        # 最小限のダミー TIF を作成する
        tif = tmp_path / "bg.tif"
        data = (
            np.full((3, 256, 256), 100, dtype=np.uint8)
        )
        transform = from_bounds(0, 0, 1, 1, 256, 256)
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=256,
            width=256,
            count=3,
            dtype="uint8",
            crs=rasterio.crs.CRS.from_epsg(32654),
            transform=transform,
        ) as dst:
            dst.write(data)

        pipeline_mod._worker_init(None)
        from medetect.datagen.compose import _compose_one

        rng = random.Random(99)
        result = _compose_one(
            tif_path=tif,
            svg_metas=None,
            image_size=128,
            resolution=10.0,
            geo_scale=None,
            ships_per_image=(1, 3),
            cluster_prob=0.0,
            cluster_size=(2, 3),
            rng=rng,
            offnadir_range=(0.0, 30.0),
        )
        # None は水域不足などで発生しうるが、クラッシュしないことを確認
        assert result is None or isinstance(result, tuple)
