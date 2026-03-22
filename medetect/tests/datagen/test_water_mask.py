from __future__ import annotations

import numpy as np
import pytest

from medetect.datagen.water_mask import (
    erode_mask,
    make_water_mask_from_rgb,
    make_water_mask_from_scl,
)


class TestMakeWaterMaskFromScl:
    def test_scl_6_is_water(self) -> None:
        """SCL値6（水域）がTrueとなる。"""
        scl = np.array([[6, 4], [5, 6]], dtype=np.uint8)
        mask = make_water_mask_from_scl(scl)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(mask, expected)

    def test_no_data_excluded(self) -> None:
        """SCL値0（NoData）は水域に含まれない。"""
        scl = np.zeros((3, 3), dtype=np.uint8)
        mask = make_water_mask_from_scl(scl)
        assert not mask.any()

    def test_cloud_excluded(self) -> None:
        """雲のSCL値(7,8,9,10)は水域に含まない。"""
        scl = np.array([7, 8, 9, 10], dtype=np.uint8).reshape(2, 2)
        mask = make_water_mask_from_scl(scl)
        assert not mask.any()

    def test_all_water(self) -> None:
        """全域水域のケース。"""
        scl = np.full((5, 5), 6, dtype=np.uint8)
        mask = make_water_mask_from_scl(scl)
        assert mask.all()


class TestErodeMask:
    def test_no_erosion_returns_same(self) -> None:
        """侵食0のとき元のマスクがそのまま返る。"""
        mask = np.ones((5, 5), dtype=bool)
        result = erode_mask(mask, 0)
        np.testing.assert_array_equal(result, mask)

    def test_erosion_shrinks_boundary(self) -> None:
        """侵食が水域マスクの境界を削る。"""
        mask = np.zeros((9, 9), dtype=bool)
        mask[1:8, 1:8] = True  # Water with land border
        result = erode_mask(mask, 1)
        # Outermost water row should be eroded
        assert not result[1, 1]
        # Centre should remain
        assert result[4, 4]

    def test_small_region_erased(self) -> None:
        """小さな水域は侵食で消える。"""
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True  # Single pixel
        result = erode_mask(mask, 1)
        assert not result.any()


class TestMakeWaterMaskFromRgb:
    def test_dark_pixels_classified_as_water(self) -> None:
        """暗いピクセルが水域として分類される。"""
        # Dark blueish pixels
        rgb = np.array([[[10, 20, 40]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert mask[0, 0]

    def test_bright_pixels_excluded(self) -> None:
        """明るいピクセルは水域から除外される。"""
        rgb = np.array([[[200, 190, 180]]], dtype=np.uint8)
        mask = make_water_mask_from_rgb(rgb)
        assert not mask[0, 0]
