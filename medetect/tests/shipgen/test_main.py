from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from medetect.shipgen.__main__ import main


class TestMainOverride:
    def test_error_when_output_dir_exists_without_override(self, tmp_path: Path) -> None:
        """--output_dir が既に存在し --override なしだとエラー終了する。"""
        out = tmp_path / "out"
        out.mkdir()

        args = [
            "--output_dir", str(out),
            "--count", "1",
            "--types", "patrol:1",
        ]
        with patch("sys.argv", ["shipgen"] + args):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2

    def test_override_removes_and_recreates_output_dir(self, tmp_path: Path) -> None:
        """--override があれば既存の --output_dir を削除して再作成する。"""
        out = tmp_path / "out"
        out.mkdir()
        (out / "old_file.svg").write_text("old")

        args = [
            "--output_dir", str(out),
            "--count", "2",
            "--types", "patrol:1",
            "--seed", "42",
            "--override",
        ]
        with patch("sys.argv", ["shipgen"] + args):
            main()

        files = list(out.glob("*.svg"))
        assert len(files) == 2
        assert not (out / "old_file.svg").exists()

    def test_no_error_when_output_dir_does_not_exist(self, tmp_path: Path) -> None:
        """--output_dir が存在しない場合は --override なしでも正常終了する。"""
        out = tmp_path / "new_dir"

        args = [
            "--output_dir", str(out),
            "--count", "1",
            "--types", "patrol:1",
            "--seed", "0",
        ]
        with patch("sys.argv", ["shipgen"] + args):
            main()

        assert out.is_dir()
        assert len(list(out.glob("*.svg"))) == 1


class TestMainFiletype:
    def test_default_filetype_is_svg(self, tmp_path: Path) -> None:
        """デフォルトの出力形式は SVG である。"""
        out = tmp_path / "out"
        args = [
            "--output_dir", str(out),
            "--count", "2",
            "--types", "patrol:1",
            "--seed", "1",
        ]
        with patch("sys.argv", ["shipgen"] + args):
            main()

        assert len(list(out.glob("*.svg"))) == 2
        assert not list(out.glob("*.png"))

    def test_filetype_png_produces_png_files(self, tmp_path: Path) -> None:
        """--filetype png で PNG ファイルが出力される。"""
        out = tmp_path / "out"
        args = [
            "--output_dir", str(out),
            "--count", "2",
            "--types", "patrol:1",
            "--seed", "1",
            "--filetype", "png",
        ]
        with patch("sys.argv", ["shipgen"] + args):
            main()

        png_files = list(out.glob("*.png"))
        assert len(png_files) == 2
        assert not list(out.glob("*.svg"))
        for f in png_files:
            assert f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_invalid_filetype_exits_with_error(self, tmp_path: Path) -> None:
        """無効な --filetype 値は exit code 2 でエラー終了する。"""
        out = tmp_path / "out"
        args = [
            "--output_dir", str(out),
            "--count", "1",
            "--filetype", "bmp",
        ]
        with patch("sys.argv", ["shipgen"] + args):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 2
