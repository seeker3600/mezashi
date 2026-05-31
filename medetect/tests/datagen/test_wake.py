"""Wake trail generation のテスト。"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from medetect.datagen.wake import (
    MotionState,
    WakePattern,
    _build_1d_noise,
    _build_noise_field,
    _make_trail_color,
    _make_wake_color,
    _pick_pattern,
    _sample_water_color,
    pick_motion_state,
    render_wake,
)


class TestPickMotionState:
    def test_returns_valid_state(self) -> None:
        """pick_motion_state は必ず MotionState を返す。"""
        rng = random.Random(0)
        for _ in range(20):
            state = pick_motion_state(rng)
            assert isinstance(state, MotionState)

    def test_all_states_occur(self) -> None:
        """十分な試行で全4状態が出現する。"""
        rng = random.Random(42)
        seen = {pick_motion_state(rng) for _ in range(400)}
        assert seen == set(MotionState)

    def test_distribution_roughly_correct(self) -> None:
        """SLOW が最多、FAST が最少になおよその分布を確認する。"""
        rng = random.Random(1)
        counts: dict[MotionState, int] = {s: 0 for s in MotionState}
        n = 2000
        for _ in range(n):
            counts[pick_motion_state(rng)] += 1
        # SLOW (40 % weight) should be most frequent
        assert counts[MotionState.SLOW] > counts[MotionState.FAST]
        assert counts[MotionState.MEDIUM] > counts[MotionState.FAST]


class TestSampleWaterColor:
    def test_returns_float32(self) -> None:
        """サンプル結果の dtype が float32 であること。"""
        bg = np.full((64, 64, 3), 80, dtype=np.uint8)
        result = _sample_water_color(bg, 32.0, 32.0)
        assert result.dtype == np.float32
        assert result.shape == (3,)

    def test_out_of_bounds_returns_fallback(self) -> None:
        """サンプル領域が画像外を全包囲する場合はフォールバック値を返す。"""
        bg = np.zeros((10, 10, 3), dtype=np.uint8)
        result = _sample_water_color(bg, -100.0, -100.0)
        assert result.shape == (3,)


class TestMakeWakeColor:
    def test_brighter_than_water(self) -> None:
        """航跡色は水面色より明るくなる。"""
        rng = random.Random(7)
        water = np.array([40.0, 60.0, 70.0], dtype=np.float32)
        wake = _make_wake_color(water, rng)
        assert float(wake.mean()) > float(water.mean())

    def test_within_byte_range(self) -> None:
        """結果が 0–255 に収まる。"""
        rng = random.Random(0)
        for _ in range(20):
            water = np.array(
                [rng.uniform(0, 200), rng.uniform(0, 200), rng.uniform(0, 200)],
                dtype=np.float32,
            )
            wake = _make_wake_color(water, rng)
            assert wake.min() >= 0.0
            assert wake.max() <= 255.0


class TestBuildNoiseField:
    def test_shape_matches(self) -> None:
        """ノイズフィールドの形状が入力と一致する。"""
        rng = random.Random(0)
        noise = _build_noise_field((64, 64), rng)
        assert noise.shape == (64, 64)

    def test_value_range(self) -> None:
        """ノイズ値が [0.55, 1.0] に収まる。"""
        rng = random.Random(0)
        noise = _build_noise_field((64, 64), rng)
        assert float(noise.min()) >= 0.54  # small float tolerance
        assert float(noise.max()) <= 1.01


class TestRenderWake:
    def _make_background(self, size: int = 128) -> np.ndarray:
        """一様な海面色の背景を生成する。"""
        return np.full((size, size, 3), 60, dtype=np.uint8)

    def _all_water_mask(self, size: int = 128) -> np.ndarray:
        return np.ones((size, size), dtype=bool)

    def test_no_op_when_alpha_scale_zero(self) -> None:
        """wake_alpha_scale=0 のとき背景は変化しない。"""
        bg = self._make_background()
        original = bg.copy()
        mask = self._all_water_mask()
        rng = random.Random(0)
        render_wake(
            bg, mask, 64.0, 64.0, 10, 30,
            angle_rad=0.0, state=MotionState.FAST, rng=rng,
            wake_alpha_scale=0.0,
        )
        np.testing.assert_array_equal(bg, original)

    def test_no_op_when_prob_scale_zero(self) -> None:
        """wake_prob_scale=0 のとき背景は変化しない。"""
        bg = self._make_background()
        original = bg.copy()
        mask = self._all_water_mask()
        for seed in range(50):
            rng = random.Random(seed)
            bg = original.copy()
            render_wake(
                bg, mask, 64.0, 64.0, 10, 30,
                angle_rad=0.0, state=MotionState.FAST, rng=rng,
                wake_prob_scale=0.0,
            )
            np.testing.assert_array_equal(
                bg, original,
                err_msg=f"Wake appeared with wake_prob_scale=0.0 at seed={seed}",
            )

    def test_prob_scale_high_increases_frequency(self) -> None:
        """wake_prob_scale > 1 のとき、デフォルトより高い頻度で航跡が出現する。"""
        mask = self._all_water_mask()
        changed_default, changed_boosted = 0, 0
        n = 100
        for i in range(n):
            bg_d = self._make_background()
            bg_b = self._make_background()
            render_wake(bg_d, mask, 64.0, 40.0, 8, 24,
                        angle_rad=0.0, state=MotionState.STOPPED,
                        rng=random.Random(i), wake_prob_scale=1.0)
            render_wake(bg_b, mask, 64.0, 40.0, 8, 24,
                        angle_rad=0.0, state=MotionState.STOPPED,
                        rng=random.Random(i), wake_prob_scale=5.0)
            if not np.array_equal(bg_d, self._make_background()):
                changed_default += 1
            if not np.array_equal(bg_b, self._make_background()):
                changed_boosted += 1
        assert changed_boosted >= changed_default, (
            f"boosted={changed_boosted} should be >= default={changed_default}"
        )

    def test_fast_ship_mostly_produces_wake(self) -> None:
        """FAST 状態の船はほぼ毎回航跡が生成される（確率 0.9）。"""
        mask = self._all_water_mask()
        changed = 0
        n = 50
        for i in range(n):
            bg = self._make_background()
            original = bg.copy()
            rng = random.Random(i)
            render_wake(
                bg, mask, 64.0, 40.0, 8, 24,
                angle_rad=0.0, state=MotionState.FAST, rng=rng,
            )
            if not np.array_equal(bg, original):
                changed += 1
        # With p=0.9, expect change ≥ 35/50 iterations.
        assert changed >= 35, f"Only {changed}/{n} trials produced a wake"

    def test_stopped_rarely_produces_wake(self) -> None:
        """STOPPED 状態の船はほとんど航跡を生成しない（確率 0.2）。"""
        mask = self._all_water_mask()
        changed = 0
        n = 50
        for i in range(n):
            bg = self._make_background()
            original = bg.copy()
            rng = random.Random(i * 3)
            render_wake(
                bg, mask, 64.0, 40.0, 8, 24,
                angle_rad=0.0, state=MotionState.STOPPED, rng=rng,
            )
            if not np.array_equal(bg, original):
                changed += 1
        # With p=0.20, expect change ≤ 20/50 trials with high probability.
        assert changed <= 25, f"{changed}/{n} trials changed (expected ≤ 25)"

    def test_wake_behind_ship_not_in_front(self) -> None:
        """航跡は船の後方にのみ現れ、船首側には出現しない。

        angle_rad=0 のとき、船首は上方（Y小）、船尾は下方（Y大）。
        航跡は Y > cy の領域（下方 = 後方）にのみ存在すべき。
        """
        size = 256
        bg_before = np.full((size, size, 3), 60, dtype=np.uint8)
        bg_after = bg_before.copy()
        mask = np.ones((size, size), dtype=bool)
        cy = 80  # ship center near top so wake (below) stays on canvas
        # Use a seed for which FAST ship definitely generates a wake
        for seed in range(30):
            rng = random.Random(seed)
            bg_after = bg_before.copy()
            render_wake(
                bg_after, mask, 128.0, float(cy), 10, 30,
                angle_rad=0.0, state=MotionState.FAST, rng=rng,
            )
            if not np.array_equal(bg_after, bg_before):
                diff = (bg_after.astype(int) - bg_before.astype(int))
                pixel_changed = (np.abs(diff).sum(axis=2) > 0)
                # No changes should occur more than one beam-width ABOVE the ship center
                upper_rows = pixel_changed[:max(0, cy - 20), :]
                assert not upper_rows.any(), (
                    f"Wake appeared in front of ship (rows above cy={cy}) with seed={seed}"
                )
                break  # one confirmed check is enough

    def test_land_mask_suppresses_wake(self) -> None:
        """陸地マスクが True の場所（水域のみ）にだけ航跡が描画される。"""
        size = 128
        bg = np.full((size, size, 3), 60, dtype=np.uint8)
        original = bg.copy()
        # Only lower half is water
        mask = np.zeros((size, size), dtype=bool)
        mask[64:, :] = True

        # angle_rad = 0: wake goes downward from stern (already in the water half)
        # Place ship center near top so the bow area upper half is all land
        for seed in range(30):
            rng = random.Random(seed)
            bg = original.copy()
            render_wake(
                bg, mask, 64.0, 20.0, 10, 30,
                angle_rad=0.0, state=MotionState.FAST, rng=rng,
            )
            if not np.array_equal(bg, original):
                diff = np.abs(bg.astype(int) - original.astype(int)).sum(axis=2)
                # Upper half (land) must remain unchanged
                assert diff[:64, :].max() == 0, "Wake appeared on land"
                break

    def test_occlusion_mask_suppresses_wake(self) -> None:
        """occlusion_mask=True の画素には航跡が描画されない。"""
        size = 128
        original = self._make_background(size)
        water_mask = self._all_water_mask(size)
        occlusion_mask = np.ones((size, size), dtype=bool)

        for seed in range(30):
            baseline = original.copy()
            render_wake(
                baseline,
                water_mask,
                64.0,
                40.0,
                8,
                24,
                angle_rad=0.0,
                state=MotionState.FAST,
                rng=random.Random(seed),
            )
            if np.array_equal(baseline, original):
                continue

            blocked = original.copy()
            render_wake(
                blocked,
                water_mask,
                64.0,
                40.0,
                8,
                24,
                angle_rad=0.0,
                state=MotionState.FAST,
                rng=random.Random(seed),
                occlusion_mask=occlusion_mask,
            )
            np.testing.assert_array_equal(blocked, original)
            break
        else:
            raise AssertionError("No seed produced a wake to validate occlusion masking")

    def test_background_dtype_preserved(self) -> None:
        """背景 ndarray の dtype が uint8 のまま保たれる。"""
        bg = self._make_background()
        mask = self._all_water_mask()
        rng = random.Random(99)
        render_wake(
            bg, mask, 64.0, 64.0, 8, 24,
            angle_rad=math.pi / 4, state=MotionState.MEDIUM, rng=rng,
        )
        assert bg.dtype == np.uint8

    def test_various_angles(self) -> None:
        """複数の角度で例外なく実行できる。"""
        mask = self._all_water_mask(256)
        for angle_deg in [0, 45, 90, 135, 180, 270]:
            bg = self._make_background(256)
            rng = random.Random(angle_deg)
            render_wake(
                bg, mask, 128.0, 128.0, 10, 30,
                angle_rad=math.radians(angle_deg),
                state=MotionState.MEDIUM, rng=rng,
            )


class TestPickPattern:
    def test_returns_valid_pattern(self) -> None:
        """_pick_pattern は WakePattern を返す。"""
        rng = random.Random(0)
        for state in MotionState:
            p = _pick_pattern(state, rng)
            assert isinstance(p, WakePattern)

    def test_stopped_mostly_foam_only(self) -> None:
        """STOPPED 状態では FOAM_ONLY が最多。"""
        rng = random.Random(42)
        counts = {p: 0 for p in WakePattern}
        for _ in range(500):
            counts[_pick_pattern(MotionState.STOPPED, rng)] += 1
        assert counts[WakePattern.FOAM_ONLY] > counts[WakePattern.FOAM_TRAIL]
        assert counts[WakePattern.FOAM_ONLY] > counts[WakePattern.FOAM_TRAIL_SPREAD]

    def test_fast_mostly_trail_or_spread(self) -> None:
        """FAST 状態では FOAM_ONLY はほとんど選ばれない。"""
        rng = random.Random(42)
        foam_count = sum(
            _pick_pattern(MotionState.FAST, rng) == WakePattern.FOAM_ONLY
            for _ in range(500)
        )
        assert foam_count < 50  # expect ~5 % × 500 = 25


class TestMakeTrailColor:
    def test_brighter_variant(self) -> None:
        """darker=False のとき水面色より明るくなる。"""
        rng = random.Random(7)
        water = np.array([40.0, 60.0, 70.0], dtype=np.float32)
        trail = _make_trail_color(water, rng, darker=False)
        assert float(trail.mean()) > float(water.mean())

    def test_darker_variant(self) -> None:
        """darker=True のとき水面色より暗くなる。"""
        rng = random.Random(7)
        water = np.array([40.0, 60.0, 70.0], dtype=np.float32)
        trail = _make_trail_color(water, rng, darker=True)
        assert float(trail.mean()) < float(water.mean())

    def test_within_byte_range(self) -> None:
        """結果が 0–255 に収まる。"""
        rng = random.Random(0)
        for _ in range(20):
            water = np.array(
                [rng.uniform(0, 200), rng.uniform(0, 200), rng.uniform(0, 200)],
                dtype=np.float32,
            )
            for darker in (True, False):
                trail = _make_trail_color(water, rng, darker=darker)
                assert trail.min() >= 0.0
                assert trail.max() <= 255.0


class TestBuild1dNoise:
    def test_shape(self) -> None:
        """長さが n_steps と一致する。"""
        rng = random.Random(0)
        noise = _build_1d_noise(100, rng)
        assert noise.shape == (100,)

    def test_value_range(self) -> None:
        """値が [0.15, 1.0] に収まる。"""
        rng = random.Random(0)
        noise = _build_1d_noise(200, rng)
        assert float(noise.min()) >= 0.14
        assert float(noise.max()) <= 1.01

    def test_not_constant(self) -> None:
        """ノイズが一定値ではない（変動がある）。"""
        rng = random.Random(42)
        noise = _build_1d_noise(100, rng)
        assert float(noise.std()) > 0.05
