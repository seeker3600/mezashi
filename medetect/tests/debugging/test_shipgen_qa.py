"""Tests for medetect.debugging.shipgen_qa."""

from __future__ import annotations

from pathlib import Path

from medetect.debugging.shipgen_qa import DEFAULT_SHIPGEN_QA_CLASSES, run_shipgen_profile_qa


class TestShipgenQa:
    def test_default_ship_set_has_ten_unique_classes(self) -> None:
        """標準 QA 対象は 10 隻で重複しない。"""
        assert len(DEFAULT_SHIPGEN_QA_CLASSES) == 10
        assert len(set(DEFAULT_SHIPGEN_QA_CLASSES)) == 10

    def test_run_shipgen_profile_qa_writes_artifacts(self, tmp_path: Path) -> None:
        """QA 実行で画像・プロファイル・マニフェストが出力される。"""
        result = run_shipgen_profile_qa(
            tmp_path,
            ship_classes=("destroyer", "carrier", "tug_harbor"),
            beam_px=96,
            length_px=480,
        )

        assert len(result.records) == 3
        assert Path(result.manifest_path).exists()
        assert Path(result.summary_path).exists()
        assert not result.offenders
        for record in result.records:
            assert Path(record.image_path).exists()
            assert Path(record.profile_tsv_path).exists()
            assert Path(record.profile_png_path).exists()