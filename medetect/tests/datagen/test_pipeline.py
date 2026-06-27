from __future__ import annotations

import pathlib
import random

import numpy as np
import pytest
from PIL import Image

import medetect.datagen.pipeline as pipeline_mod

from medetect.datagen.pipeline import _false_source_grid, _write_dataset_yaml, generate_false_negatives
from medetect.datagen.scene import DEFAULT_EDGE_HARDNESS


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
        """しきい値なしのとき、ship / ship_c の2クラスを出力する。"""
        _write_dataset_yaml(tmp_path, 0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  0: ship\n" in content
        assert "  1: ship_c\n" in content

    def test_two_classes_with_threshold(self, tmp_path: pathlib.Path) -> None:
        """しきい値ありのとき、solo 2クラス + cluster 2クラスを出力する。"""
        _write_dataset_yaml(tmp_path, 0, size_thresholds=(100.0,))
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  0: ship_small\n" in content
        assert "  1: ship_large\n" in content
        assert "  2: ship_small_c\n" in content
        assert "  3: ship_large_c\n" in content

    def test_three_classes_with_two_thresholds(self, tmp_path: pathlib.Path) -> None:
        """しきい値2つのとき、solo 3クラス + cluster 3クラスを出力する。"""
        _write_dataset_yaml(tmp_path, 0, size_thresholds=(30.0, 80.0))
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  0: ship_small\n" in content
        assert "  1: ship_medium\n" in content
        assert "  2: ship_large\n" in content
        assert "  3: ship_small_c\n" in content
        assert "  4: ship_medium_c\n" in content
        assert "  5: ship_large_c\n" in content

    def test_four_classes_with_three_thresholds(self, tmp_path: pathlib.Path) -> None:
        """しきい値3つのとき、境界値を使った solo+cluster 各4クラスを出力する。"""
        _write_dataset_yaml(tmp_path, 0, size_thresholds=(20.0, 50.0, 100.0))
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  0: ship_small\n" in content
        assert "  1: ship_20_50\n" in content
        assert "  2: ship_50_100\n" in content
        assert "  3: ship_large\n" in content
        assert "  4: ship_small_c\n" in content
        assert "  5: ship_20_50_c\n" in content
        assert "  6: ship_50_100_c\n" in content
        assert "  7: ship_large_c\n" in content

    def test_params_written_as_comments(self, tmp_path: pathlib.Path) -> None:
        """生成パラメータがコメントとして書き込まれる。"""
        params = {"count": 100, "resolution": 10.0, "size_thresholds": 80.0}
        _write_dataset_yaml(tmp_path, 0, size_thresholds=(80.0,), params=params)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "# Generation parameters:" in content
        assert "#   count: 100" in content
        assert "#   resolution: 10.0" in content
        assert "#   size_thresholds: 80.0" in content

    def test_no_params_no_comment(self, tmp_path: pathlib.Path) -> None:
        """パラメータなしのとき、コメント行がない。"""
        _write_dataset_yaml(tmp_path, 0)
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "# Generation parameters:" not in content

    def test_custom_class_id_with_threshold(self, tmp_path: pathlib.Path) -> None:
        """class_id が 0 以外でも正しい ID で出力される。"""
        _write_dataset_yaml(tmp_path, 3, size_thresholds=(50.0,))
        content = (tmp_path / "dataset.yaml").read_text(encoding="utf-8")
        assert "  3: ship_small\n" in content
        assert "  4: ship_large\n" in content
        assert "  5: ship_small_c\n" in content
        assert "  6: ship_large_c\n" in content


class TestGenerateDatasetParams:
    """generate_dataset の記録パラメータ整合を検証する。"""

    def test_debug_params_are_not_written(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """デバッグ系パラメータは dataset.yaml 用 params に含めない。"""
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
            size_thresholds: tuple[float, ...] | None = None,
            params: dict[str, object] | None = None,
        ) -> None:
            captured["params"] = dict(params or {})

        monkeypatch.setattr(
            pipeline_mod.concurrent.futures,
            "ProcessPoolExecutor",
            _DummyExecutor,
        )
        monkeypatch.setattr(pipeline_mod, "_write_dataset_yaml", _capture_yaml)

        pipeline_mod.generate_dataset(
            bg_dir=bg_dir,
            output_dir=tmp_path / "out",
            count=0,
            debug_bg_color=(0x12, 0x34, 0x56),
            max_workers=1,
        )

        params = captured["params"]
        assert isinstance(params, dict)
        assert "force_tight_clusters" not in params
        assert "debug_bg_color" not in params
        assert "disable_water_tint" not in params
        assert "water_tint_strength" not in params
        assert "cluster_blend_strength" not in params
        assert params["edge_hardness"] == pytest.approx(DEFAULT_EDGE_HARDNESS)
        assert params["shadow_alpha_scale"] == 1.0
        assert params["shadow_length_range"] == "0.0:3.75"

    def test_workers_zero_runs_without_process_pool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """max_workers=0 のとき ProcessPoolExecutor を使わず逐次実行する。"""
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        for index in range(2):
            (bg_dir / f"scene_{index}_visual.tif").write_bytes(b"placeholder")

        calls: list[tuple[int, pathlib.Path, tuple[int, int, int] | None, float]] = []
        init_calls: list[tuple[pathlib.Path | None, pathlib.Path | None]] = []

        def _fail_executor(*args, **kwargs):
            msg = "workers=0 should not create ProcessPoolExecutor"
            raise AssertionError(msg)

        def _capture_worker_init(
            svg_dir: pathlib.Path | None,
            coastline_path: pathlib.Path | None = None,
        ) -> None:
            init_calls.append((svg_dir, coastline_path))

        def _capture_compose_task(
            *,
            index: int,
            task_seed: int,
            tif_path: pathlib.Path | None,
            img_out: pathlib.Path,
            lbl_out: pathlib.Path,
            config: object,
            expected_surface: str | None = None,
            candidate_tifs: tuple[pathlib.Path, ...] = (),
            surface_target_attempts: int = 12,
        ) -> tuple[int, int, str]:
            del task_seed, img_out, lbl_out, candidate_tifs, surface_target_attempts
            calls.append(
                (
                    index,
                    tif_path,
                    getattr(config, "debug_bg_color"),
                    getattr(config, "edge_hardness"),
                )
            )
            assert expected_surface is None
            return 1, 0, "mixed"

        monkeypatch.setattr(
            pipeline_mod.concurrent.futures,
            "ProcessPoolExecutor",
            _fail_executor,
        )
        monkeypatch.setattr(pipeline_mod, "_worker_init", _capture_worker_init)
        monkeypatch.setattr(pipeline_mod, "_run_compose_task", _capture_compose_task)
        monkeypatch.setattr(pipeline_mod, "_write_dataset_yaml", lambda *args, **kwargs: None)

        stats = pipeline_mod.generate_dataset(
            bg_dir=bg_dir,
            output_dir=tmp_path / "out",
            count=2,
            debug_bg_color=(1, 2, 3),
            edge_hardness=0.3,
            max_workers=0,
        )

        assert len(init_calls) == 1
        assert len(calls) == 2
        assert {call[0] for call in calls} == {0, 1}
        assert all(call[2] == (1, 2, 3) for call in calls)
        assert all(call[3] == pytest.approx(0.3) for call in calls)
        assert stats["images"] == 2
        assert stats["ships"] == 2
        assert stats["clusters"] == 0

    def test_berth_params_are_forwarded_to_config_and_yaml(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """berth 系パラメータは compose config と dataset.yaml params に残る。"""
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        (bg_dir / "scene_visual.tif").write_bytes(b"placeholder")

        captured: dict[str, object] = {}
        seen_config: list[object] = []

        def _capture_yaml(
            output_dir: pathlib.Path,
            class_id: int,
            *,
            size_thresholds: tuple[float, ...] | None = None,
            params: dict[str, object] | None = None,
        ) -> None:
            del output_dir, class_id, size_thresholds
            captured["params"] = dict(params or {})

        def _capture_compose_task(
            *,
            index: int,
            task_seed: int,
            tif_path: pathlib.Path | None,
            img_out: pathlib.Path,
            lbl_out: pathlib.Path,
            config: object,
            expected_surface: str | None = None,
            candidate_tifs: tuple[pathlib.Path, ...] = (),
            surface_target_attempts: int = 12,
        ) -> tuple[int, int, str]:
            del index, task_seed, tif_path, img_out, lbl_out, candidate_tifs, surface_target_attempts
            seen_config.append(config)
            assert expected_surface is None
            return 0, 0, "mixed"

        monkeypatch.setattr(pipeline_mod, "_write_dataset_yaml", _capture_yaml)
        monkeypatch.setattr(pipeline_mod, "_run_compose_task", _capture_compose_task)
        monkeypatch.setattr(pipeline_mod, "_worker_init", lambda *args, **kwargs: None)

        pipeline_mod.generate_dataset(
            bg_dir=bg_dir,
            output_dir=tmp_path / "out",
            count=1,
            berth_prob=0.8,
            berth_stern_prob=0.2,
            max_workers=0,
        )

        assert len(seen_config) == 1
        assert seen_config[0].berth_prob == pytest.approx(0.8)
        assert seen_config[0].berth_stern_prob == pytest.approx(0.2)
        params = captured["params"]
        assert isinstance(params, dict)
        assert params["berth_prob"] == pytest.approx(0.8)
        assert params["berth_stern_prob"] == pytest.approx(0.2)
        assert params["berth_cluster_auto_truncate"] is True

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


class TestBackgroundSurfaceMixRatio:
    """合成背景比率オプションの適用とフォールバックを検証する。"""

    def test_applies_to_synth_remainder_after_false_ratio(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """false_ratio 後の synth_count 件にだけ surface 比率が適用される。"""
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        (bg_dir / "scene_visual.tif").write_bytes(b"placeholder")

        constrained_calls: list[int] = []

        def _capture_compose_task(
            *,
            index: int,
            task_seed: int,
            tif_path: pathlib.Path | None,
            img_out: pathlib.Path,
            lbl_out: pathlib.Path,
            config: object,
            expected_surface: str | None = None,
            candidate_tifs: tuple[pathlib.Path, ...] = (),
            surface_target_attempts: int = 12,
        ) -> tuple[int, int, str]:
            del task_seed, tif_path, img_out, lbl_out, config, candidate_tifs, surface_target_attempts
            if expected_surface is not None:
                constrained_calls.append(index)
                assert expected_surface == "sea_only"
            return 1, 0, "sea_only"

        monkeypatch.setattr(pipeline_mod, "_run_compose_task", _capture_compose_task)
        monkeypatch.setattr(pipeline_mod, "_write_dataset_yaml", lambda *args, **kwargs: None)
        monkeypatch.setattr(pipeline_mod, "_worker_init", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            pipeline_mod,
            "generate_false_negatives",
            lambda *args, **kwargs: 3,
        )

        stats = pipeline_mod.generate_dataset(
            bg_dir=bg_dir,
            output_dir=tmp_path / "out",
            count=10,
            false_dir=tmp_path,
            false_ratio=0.3,
            bg_surface_mix_ratio=(1.0, 0.0),
            max_workers=0,
        )

        # count=10, false_ratio=0.3 -> synth_count=7
        assert sorted(constrained_calls) == list(range(7))
        assert stats["images"] == 7
        assert stats["false_negatives"] == 3

    def test_unmet_target_falls_back_to_unconstrained(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """カテゴリ未達時は同じ index を制約なしで再試行して埋める。"""
        bg_dir = tmp_path / "bg"
        bg_dir.mkdir()
        (bg_dir / "scene_visual.tif").write_bytes(b"placeholder")

        retry_counts: dict[int, int] = {}

        def _capture_compose_task(
            *,
            index: int,
            task_seed: int,
            tif_path: pathlib.Path | None,
            img_out: pathlib.Path,
            lbl_out: pathlib.Path,
            config: object,
            expected_surface: str | None = None,
            candidate_tifs: tuple[pathlib.Path, ...] = (),
            surface_target_attempts: int = 12,
        ) -> tuple[int, int, str]:
            del task_seed, tif_path, img_out, lbl_out, config, candidate_tifs
            retry_counts[index] = surface_target_attempts
            return 1, 0, "mixed"

        monkeypatch.setattr(pipeline_mod, "_run_compose_task", _capture_compose_task)
        monkeypatch.setattr(pipeline_mod, "_write_dataset_yaml", lambda *args, **kwargs: None)
        monkeypatch.setattr(pipeline_mod, "_worker_init", lambda *args, **kwargs: None)

        stats = pipeline_mod.generate_dataset(
            bg_dir=bg_dir,
            output_dir=tmp_path / "out",
            count=5,
            bg_surface_mix_ratio=(1.0, 0.0),
            max_workers=0,
        )

        assert stats["images"] == 5
        assert stats["surface_goal_rejected"] == 5
        assert stats["mixed"] == 5
        assert all(retry_counts[index] == 12 for index in range(5))


class TestRunComposeTaskSurfaceTarget:
    """_run_compose_task の surface target retry を検証する。"""

    def test_retries_until_expected_surface_matches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """expected_surface 指定時は一致するまで複数回試行する。"""
        import medetect.datagen.compose as compose_mod

        image_out = tmp_path / "images"
        label_out = tmp_path / "labels"
        image_out.mkdir()
        label_out.mkdir()

        tif_a = tmp_path / "a.tif"
        tif_b = tmp_path / "b.tif"
        tif_a.write_bytes(b"a")
        tif_b.write_bytes(b"b")

        attempts: list[pathlib.Path] = []

        def _fake_compose_one_with_surface_category(**kwargs):
            tif_path = kwargs["tif_path"]
            attempts.append(tif_path)
            tile = np.zeros((32, 32, 3), dtype=np.uint8)
            if len(attempts) < 3:
                return tile, [], 0, "sea_only"
            return tile, [], 0, "mixed"

        monkeypatch.setattr(
            compose_mod,
            "_compose_one_with_surface_category",
            _fake_compose_one_with_surface_category,
        )

        config = pipeline_mod._ComposeTaskConfig(
            image_size=32,
            resolution=10.0,
            geo_scale=1.0,
            ships_per_image=(0, 0),
            cluster_prob=0.0,
            cluster_size=(2, 2),
            cluster_mixed_prob=0.5,
            class_id=0,
            erode_coast=0,
            min_water_ratio=0.0,
            edge_hardness=0.75,
            ship_alpha=(0.7, 1.0),
            ship_length_range=None,
            length_exponent=1.0,
            berth_prob=0.0,
            berth_stern_prob=0.0,
            size_thresholds=None,
            wake_prob_scale=1.0,
            wake_alpha_scale=1.0,
            debug_bg_color=None,
            shadow_alpha_scale=1.0,
            shadow_length_range=(0.0, 0.0),
            offnadir_range=(0.0, 0.0),
            shipgen_kwargs={},
        )

        result = pipeline_mod._run_compose_task(
            index=0,
            task_seed=0,
            tif_path=tif_a,
            img_out=image_out,
            lbl_out=label_out,
            config=config,
            expected_surface="mixed",
            candidate_tifs=(tif_a, tif_b),
            surface_target_attempts=4,
        )

        assert result == (0, 0, "mixed")
        assert len(attempts) == 3
        assert (image_out / "000000.png").exists()
        assert (label_out / "000000.txt").exists()
        with Image.open(image_out / "000000.png") as output:
            assert output.size == (32, 32)


class TestDatagenCli:
    """datagen CLI の公開オプション整合を検証する。"""

    def test_help_reflects_current_debug_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--help に現行のデバッグ系オプションが反映される。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        monkeypatch.setattr(sys, "argv", ["medetect.datagen", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            datagen_main.main()

        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out

        assert "--force_tight_clusters" not in help_text
        assert "--debug_bg_color" in help_text
        assert "--edge_hardness" in help_text
        assert "--water_tint_strength" not in help_text
        assert "--cluster_blend_strength" not in help_text
        assert "--disable-water-tint" not in help_text
        assert "--shadow_elevation" not in help_text
        assert "--bg_surface_mix_ratio" in help_text
        assert "placement events per image" in help_text
        assert "single ships only" in help_text
        assert "reusing the same ship" in help_text
        assert "ship uniform" in help_text

    def test_debug_bg_color_is_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """debug_bg_color は #RRGGBB から RGB タプルへ変換されて渡る。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        captured: dict[str, object] = {}

        def _capture_generate_dataset(**kwargs):
            captured.update(kwargs)
            return {"images": 0, "ships": 0, "clusters": 0, "skipped": 0}

        monkeypatch.setattr(datagen_main, "generate_dataset", _capture_generate_dataset)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect.datagen",
                "--bg_dir",
                "bg",
                "--output_dir",
                "out",
                "--count",
                "1",
                "--debug_bg_color",
                "#123456",
                "--workers",
                "0",
            ],
        )

        datagen_main.main()

        assert captured["debug_bg_color"] == (0x12, 0x34, 0x56)
        assert captured["max_workers"] == 0

    def test_edge_hardness_is_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """edge_hardness は CLI から generate_dataset へ渡る。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        captured: dict[str, object] = {}

        def _capture_generate_dataset(**kwargs):
            captured.update(kwargs)
            return {"images": 0, "ships": 0, "clusters": 0, "skipped": 0}

        monkeypatch.setattr(datagen_main, "generate_dataset", _capture_generate_dataset)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect.datagen",
                "--bg_dir",
                "bg",
                "--output_dir",
                "out",
                "--count",
                "1",
                "--edge_hardness",
                "0.25",
            ],
        )

        datagen_main.main()

        assert captured["edge_hardness"] == pytest.approx(0.25)
        assert "water_tint_strength" not in captured
        assert "cluster_blend_strength" not in captured

    def test_berth_params_are_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """berth 系パラメータは CLI から generate_dataset へ渡る。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        captured: dict[str, object] = {}

        def _capture_generate_dataset(**kwargs):
            captured.update(kwargs)
            return {"images": 0, "ships": 0, "clusters": 0, "skipped": 0}

        monkeypatch.setattr(datagen_main, "generate_dataset", _capture_generate_dataset)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect.datagen",
                "--bg_dir",
                "bg",
                "--output_dir",
                "out",
                "--count",
                "1",
                "--berth_prob",
                "0.8",
                "--berth_stern_prob",
                "0.2",
            ],
        )

        datagen_main.main()

        assert captured["berth_prob"] == pytest.approx(0.8)
        assert captured["berth_stern_prob"] == pytest.approx(0.2)

    def test_ship_lb_ratio_is_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ship_lb_ratio は CLI から generate_dataset へ渡る。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        captured: dict[str, object] = {}

        def _capture_generate_dataset(**kwargs):
            captured.update(kwargs)
            return {"images": 0, "ships": 0, "clusters": 0, "skipped": 0}

        monkeypatch.setattr(datagen_main, "generate_dataset", _capture_generate_dataset)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect.datagen",
                "--bg_dir",
                "bg",
                "--output_dir",
                "out",
                "--count",
                "1",
                "--ship_lb_ratio",
                "4.0:8.0",
            ],
        )

        datagen_main.main()

        assert captured["ship_lb_ratio_range"] == (4.0, 8.0)

    def test_ship_lb_ratio_rejects_invalid_range(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ship_lb_ratio の min > max は CLI で拒否される。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect.datagen",
                "--bg_dir",
                "bg",
                "--output_dir",
                "out",
                "--count",
                "1",
                "--ship_lb_ratio",
                "8.0:4.0",
            ],
        )

        with pytest.raises(SystemExit):
            datagen_main.main()

    def test_bg_surface_mix_ratio_is_forwarded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """bg_surface_mix_ratio は CLI から generate_dataset へ渡る。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        captured: dict[str, object] = {}

        def _capture_generate_dataset(**kwargs):
            captured.update(kwargs)
            return {"images": 0, "ships": 0, "clusters": 0, "skipped": 0}

        monkeypatch.setattr(datagen_main, "generate_dataset", _capture_generate_dataset)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect.datagen",
                "--bg_dir",
                "bg",
                "--output_dir",
                "out",
                "--count",
                "1",
                "--bg_surface_mix_ratio",
                "0.6:0.4",
            ],
        )

        datagen_main.main()

        assert captured["bg_surface_mix_ratio"] == (0.6, 0.4)

    def test_bg_surface_mix_ratio_sum_must_be_positive(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """bg_surface_mix_ratio の和が 0 以下ならエラーになる。"""
        import sys

        import medetect.datagen.__main__ as datagen_main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "medetect.datagen",
                "--bg_dir",
                "bg",
                "--output_dir",
                "out",
                "--count",
                "1",
                "--bg_surface_mix_ratio",
                "0:0",
            ],
        )

        with pytest.raises(SystemExit):
            datagen_main.main()

