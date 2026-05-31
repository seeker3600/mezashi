"""Tests for medetect.debugging.berth_preview."""

from __future__ import annotations

from pathlib import Path

import medetect.debugging.berth_preview as berth_preview_mod
from medetect.debugging.berth_preview import count_berth_seed_sweep, render_berth_previews


class TestCountBerthSeedSweep:
    def test_returns_expected_keys_and_total(self) -> None:
        """seed sweep 件数は期待キーを持ち、single+cluster の合計件数になる。"""
        counts = count_berth_seed_sweep(seed_count=8)

        assert set(counts) == {
            "single_alongside",
            "single_stern",
            "cluster_tight_alongside",
            "cluster_tight_stern",
            "fallback_open",
        }
        assert sum(counts.values()) == 16


class TestRenderBerthPreviews:
    def test_writes_seed_sweep_manifest(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """preview render は seed sweep manifest も出力する。"""

        def _write_case(
            output_dir: Path,
            name: str,
            water_mask,
            berth_segments,
            *,
            berth_stern_prob: float,
            seed: int,
        ) -> Path:
            del water_mask, berth_segments, berth_stern_prob, seed
            path = output_dir / f"{name}.png"
            path.write_bytes(b"png")
            return path

        monkeypatch.setattr(berth_preview_mod, "_write_case", _write_case)
        monkeypatch.setattr(
            berth_preview_mod,
            "count_berth_seed_sweep",
            lambda seed_count=64: {
                "single_alongside": 2,
                "single_stern": 1,
                "cluster_tight_alongside": 3,
                "cluster_tight_stern": 4,
                "fallback_open": 0,
            },
        )

        outputs = render_berth_previews(tmp_path)

        counts_path = outputs["seed_sweep_counts"]
        assert counts_path.exists()
        assert counts_path.read_text(encoding="utf-8").splitlines() == [
            "single_alongside: 2",
            "single_stern: 1",
            "cluster_tight_alongside: 3",
            "cluster_tight_stern: 4",
            "fallback_open: 0",
        ]
