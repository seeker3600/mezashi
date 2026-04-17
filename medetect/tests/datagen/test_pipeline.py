from __future__ import annotations

import pathlib
import random

import numpy as np
import pytest

import medetect.datagen.pipeline as pipeline_mod

from medetect.datagen.pipeline import _false_source_grid, _write_dataset_yaml, generate_false_negatives


class TestWorkerInit:
    """_worker_init によるワーカープロセス初期化のテスト。"""

    def test_none_svg_dir_sets_none(self) -> None:
        """svg_dir=None のとき _worker_svg_metas が None になる。"""
        pipeline_mod._worker_init(None)
        assert pipeline_mod._worker_svg_metas is None

    def test_svg_dir_populates_metas(self, tmp_path: pathlib.Path) -> None:
        """有効な SVG ディレクトリを渡すと _worker_svg_metas が設定される。"""
        from medetect.shipgen.gen import generate_ship_svg

        svg_dir = tmp_path / "svgs"
        svg_dir.mkdir()
        rng = random.Random(0)
        for index in range(3):
            (svg_dir / f"ship_{index}.svg").write_text(
                generate_ship_svg("patrol", rng=rng),
                encoding="utf-8",
            )

        pipeline_mod._worker_init(svg_dir)
        assert pipeline_mod._worker_svg_metas is not None
        assert len(pipeline_mod._worker_svg_metas) == 3
        for meta in pipeline_mod._worker_svg_metas:
            assert meta.lb_ratio > 0


class TestWriteDatasetYaml:
    """_write_dataset_yaml の出力検証。"""

    def test_single_class_without_threshold(self, tmp_path: pathlib.Path) -> None:
        """しきい値なしのとき、単一クラス ship を出力する。"""
        _write_dataset_yaml(tmp_path, 0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  0: ship\n" in content
        assert "ship_small" not in content
        assert "ship_large" not in content

    def test_two_classes_with_threshold(self, tmp_path: pathlib.Path) -> None:
        """しきい値ありのとき、2クラスを出力する。"""
        _write_dataset_yaml(tmp_path, 0, size_threshold=100.0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  0: ship_small\n" in content
        assert "  1: ship_large\n" in content

    def test_params_written_as_comments(self, tmp_path: pathlib.Path) -> None:
        """生成パラメータがコメントとして書き込まれる。"""
        params = {"count": 100, "resolution": 10.0, "size_threshold": 80.0}
        _write_dataset_yaml(tmp_path, 0, size_threshold=80.0, params=params)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "# Generation parameters:" in content
        assert "#   count: 100" in content
        assert "#   resolution: 10.0" in content
        assert "#   size_threshold: 80.0" in content

    def test_no_params_no_comment(self, tmp_path: pathlib.Path) -> None:
        """パラメータなしのとき、コメント行がない。"""
        _write_dataset_yaml(tmp_path, 0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "# Generation parameters:" not in content

    def test_custom_class_id_with_threshold(self, tmp_path: pathlib.Path) -> None:
        """class_id が 0 以外でも正しい ID で出力される。"""
        _write_dataset_yaml(tmp_path, 3, size_threshold=50.0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  3: ship_small\n" in content
        assert "  4: ship_large\n" in content


class TestGenerateDatasetParams:
    """generate_dataset の記録パラメータ整合を検証する。"""

    def test_removed_debug_params_are_not_written(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """削除済みデバッグ系パラメータは dataset.yaml 用 params に含めない。"""
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        (bg_dir / "scene_visual.tif").write_bytes(b"placeholder")

        captured: dict[str, object] = {}

        class _DummyExecutor:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def submit(self, *args, **kwargs):
                msg = "count=0 should not submit compose tasks"
                raise AssertionError(msg)

        def _capture_yaml(
            output_dir: pathlib.Path,
            class_id: int,
            *,
            size_threshold: float | None = None,
            params: dict[str, object] | None = None,
        ) -> None:
            captured["params"] = dict(params or {})

        monkeypatch.setattr(pipeline_mod.concurrent.futures, "ProcessPoolExecutor", _DummyExecutor)
        monkeypatch.setattr(pipeline_mod, "_write_dataset_yaml", _capture_yaml)

        pipeline_mod.generate_dataset(
            bg_dir=bg_dir,
            output_dir=tmp_path / "out",
            count=0,
            max_workers=1,
        )

        params = captured["params"]
        assert isinstance(params, dict)
        assert "force_tight_clusters" not in params
        assert "debug_bg_color" not in params
        assert "disable_water_tint" not in params


class TestFalseSourceGrid:
    """_false_source_grid のグリッド計算テスト。"""

    def test_png_exact_tiles(self, tmp_path: pathlib.Path) -> None:
        """PNG 画像のグリッドサイズが正しく計算される。"""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (1280, 640))
        path = tmp_path / "src.png"
        img.save(path)
        result = _false_source_grid(path, image_size=640, resolution=None, geo_scale=None)
        assert result == (640, 2, 1)

    def test_png_too_small_returns_none(self, tmp_path: pathlib.Path) -> None:
        """小さすぎる PNG は None を返す。"""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (320, 320))
        path = tmp_path / "small.png"
        img.save(path)
        result = _false_source_grid(path, image_size=640, resolution=None, geo_scale=None)
        assert result is None

    def test_partial_tile_truncated(self, tmp_path: pathlib.Path) -> None:
        """端数はタイルに含まれない。"""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (1500, 900))
        path = tmp_path / "partial.png"
        img.save(path)
        result = _false_source_grid(path, image_size=640, resolution=None, geo_scale=None)
        assert result == (640, 2, 1)

    def test_geo_scale_applied_to_tif(self, tmp_path: pathlib.Path) -> None:
        """TIFF に geo_scale が適用される。"""
        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "bg.tif"
        size = 3200
        data = np.full((3, size, size), 128, dtype=np.uint8)
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=3,
            dtype="uint8",
            transform=from_bounds(0, 0, size, size, size, size),
        ) as dst:
            dst.write(data)
        result = _false_source_grid(tif_path, image_size=640, resolution=None, geo_scale=2.0)
        assert result is not None
        src_tile, cols, rows = result
        assert src_tile == 1280
        assert cols == size // 1280
        assert rows == size // 1280


class TestGenerateFalseNegatives:
    """generate_false_negatives の機能テスト。"""

    @staticmethod
    def _make_source(path: pathlib.Path, width: int, height: int, color: tuple) -> None:
        from PIL import Image as PILImage

        PILImage.new("RGB", (width, height), color=color).save(path)

    def test_writes_images_and_empty_labels(self, tmp_path: pathlib.Path) -> None:
        """False negative タイルと空ラベルが正しく書き出される。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        self._make_source(false_dir / "a.png", 1280, 640, (80, 100, 120))
        self._make_source(false_dir / "b.png", 1280, 640, (60, 80, 100))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        n_written = generate_false_negatives(
            false_dir,
            out_dir,
            count=3,
            image_size=640,
            rng=rng,
            start_index=0,
        )
        assert n_written == 3
        for index in range(3):
            assert (out_dir / "images" / "train" / f"{index:06d}.png").exists()
            label_path = out_dir / "labels" / "train" / f"{index:06d}.txt"
            assert label_path.exists()
            assert label_path.read_text(encoding="utf-8") == ""

    def test_start_index_offsets_names(self, tmp_path: pathlib.Path) -> None:
        """start_index によりファイル番号がオフセットされる。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        self._make_source(false_dir / "a.png", 1280, 640, (80, 100, 120))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        n_written = generate_false_negatives(
            false_dir,
            out_dir,
            count=2,
            image_size=640,
            rng=rng,
            start_index=100,
        )
        assert n_written == 2
        assert (out_dir / "images" / "train" / "000100.png").exists()
        assert (out_dir / "images" / "train" / "000101.png").exists()
        assert not (out_dir / "images" / "train" / "000000.png").exists()

    def test_no_overlap_exhausts_grid(self, tmp_path: pathlib.Path) -> None:
        """1画像から全タイルを要求しても重複なく書き出せる。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        self._make_source(false_dir / "a.png", 2560, 2560, (80, 100, 120))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        n_written = generate_false_negatives(false_dir, out_dir, count=16, image_size=640, rng=rng)
        assert n_written == 16

    def test_even_distribution_multi_source(self, tmp_path: pathlib.Path) -> None:
        """複数ソース間で均等配分される。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        for index in range(4):
            self._make_source(false_dir / f"src{index}.png", 640, 640, (50 + index * 20, 60, 70))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        n_written = generate_false_negatives(false_dir, out_dir, count=4, image_size=640, rng=rng)
        assert n_written == 4

    def test_empty_dir_raises(self, tmp_path: pathlib.Path) -> None:
        """ソース画像がないディレクトリは FileNotFoundError を送出する。"""
        false_dir = tmp_path / "false"
        false_dir.mkdir()
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(42)
        with pytest.raises(FileNotFoundError):
            generate_false_negatives(false_dir, out_dir, count=2, image_size=640, rng=rng)

    def test_tile_size_matches_image_size(self, tmp_path: pathlib.Path) -> None:
        """書き出された PNG のサイズが image_size と一致する。"""
        from PIL import Image as PILImage

        false_dir = tmp_path / "false"
        false_dir.mkdir()
        self._make_source(false_dir / "a.png", 1280, 640, (80, 100, 120))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(0)
        generate_false_negatives(false_dir, out_dir, count=1, image_size=320, rng=rng)
        with PILImage.open(out_dir / "images" / "train" / "000000.png") as img:
            assert img.size == (320, 320)

    def test_count_met_when_capacity_insufficient(
        self,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """容量不足でも要求枚数分が出力され、警告が発される。"""
        import logging

        false_dir = tmp_path / "false"
        false_dir.mkdir()
        self._make_source(false_dir / "a.png", 1280, 640, (80, 100, 120))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(7)
        with caplog.at_level(logging.WARNING, logger="medetect.datagen.pipeline"):
            n_written = generate_false_negatives(false_dir, out_dir, count=5, image_size=640, rng=rng)
        assert n_written == 5
        assert (out_dir / "images" / "train" / "000004.png").exists()
        assert any("repeated" in record.message.lower() for record in caplog.records)

    def test_count_met_multi_source_capacity_insufficient(
        self,
        tmp_path: pathlib.Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """複数ソースでも容量不足時に全枚数が出力される。"""
        import logging

        false_dir = tmp_path / "false"
        false_dir.mkdir()
        self._make_source(false_dir / "a.png", 640, 640, (80, 100, 120))
        self._make_source(false_dir / "b.png", 640, 640, (60, 80, 100))
        out_dir = tmp_path / "out"
        (out_dir / "images" / "train").mkdir(parents=True)
        (out_dir / "labels" / "train").mkdir(parents=True)

        rng = random.Random(3)
        with caplog.at_level(logging.WARNING, logger="medetect.datagen.pipeline"):
            n_written = generate_false_negatives(false_dir, out_dir, count=7, image_size=640, rng=rng)
        assert n_written == 7
        assert (out_dir / "images" / "train" / "000006.png").exists()
        assert any("repeated" in record.message.lower() for record in caplog.records)


class TestFalseRatioSplit:
    """false_ratio による合成/偽陰性の枚数分割テスト。"""

    def test_ratio_zero_no_false(self) -> None:
        """false_ratio=0.0 のとき偽陰性は 0 枚。"""
        total = 10
        false_ratio = 0.0
        false_count = round(total * false_ratio)
        synth_count = total - false_count
        assert synth_count == 10
        assert false_count == 0

    def test_ratio_09_gives_correct_split(self) -> None:
        """count=100, false_ratio=0.9 の分割が正しい。"""
        total = 100
        false_ratio = 0.9
        false_count = round(total * false_ratio)
        synth_count = total - false_count
        assert synth_count == 10
        assert false_count == 90
        assert synth_count + false_count == total

    def test_ratio_02_gives_correct_split(self) -> None:
        """count=100, false_ratio=0.2 の分割が正しい。"""
        total = 100
        false_ratio = 0.2
        false_count = round(total * false_ratio)
        synth_count = total - false_count
        assert synth_count == 80
        assert false_count == 20
        assert synth_count + false_count == total

    def test_ratio_05_splits_evenly(self) -> None:
        """count=100, false_ratio=0.5 の分割が正しい。"""
        total = 100
        false_ratio = 0.5
        false_count = round(total * false_ratio)
        synth_count = total - false_count
        assert synth_count == 50
        assert false_count == 50
        assert synth_count + false_count == total

    def test_total_preserved_for_various_counts(self) -> None:
        """様々な count/ratio で合計が常に count に等しい。"""
        for total, ratio in [(10, 0.3), (7, 0.5), (1000, 0.1), (3, 0.9)]:
            false_count = round(total * ratio)
            synth_count = total - false_count
            assert synth_count + false_count == total


class TestDatagenCli:
    """datagen CLI の公開オプション整合を検証する。"""

    def test_help_omits_removed_debug_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--help から削除対象オプションが消え、説明文が現行仕様に一致する。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        monkeypatch.setattr(sys, "argv", ["medetect.datagen", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            datagen_main.main()

        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out

        assert "--force_tight_clusters" not in help_text
        assert "--debug_bg_color" not in help_text
        assert "--disable-water-tint" not in help_text
        assert "placement events per image" in help_text
        assert "single ships only" in help_text