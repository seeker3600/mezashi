from __future__ import annotations

import argparse

import pytest

from medetect.datagen.__main__ import _parse_shipgen_kwarg


class TestParseShipgenKwarg:
    def test_int_conversion(self) -> None:
        """整数値は int に変換される。"""
        key, value = _parse_shipgen_kwarg("n_hull_points=32")
        assert key == "n_hull_points"
        assert value == 32
        assert isinstance(value, int)

    def test_float_conversion(self) -> None:
        """小数値は float に変換される。"""
        key, value = _parse_shipgen_kwarg("hull_noise=0.01")
        assert key == "hull_noise"
        assert value == pytest.approx(0.01)
        assert isinstance(value, float)

    def test_string_fallback(self) -> None:
        """リテラルとして解釈できない値はそのまま str になる。"""
        key, value = _parse_shipgen_kwarg("trim_mode=bow")
        assert key == "trim_mode"
        assert value == "bow"
        assert isinstance(value, str)

    def test_none_conversion(self) -> None:
        """'None' は Python の None に変換される。"""
        key, value = _parse_shipgen_kwarg("trim_mode=None")
        assert key == "trim_mode"
        assert value is None

    def test_bool_true_conversion(self) -> None:
        """'True' は Python の True に変換される。"""
        key, value = _parse_shipgen_kwarg("some_flag=True")
        assert value is True

    def test_bool_false_conversion(self) -> None:
        """'False' は Python の False に変換される。"""
        key, value = _parse_shipgen_kwarg("some_flag=False")
        assert value is False

    def test_no_equals_raises(self) -> None:
        """= を含まない入力は ArgumentTypeError を送出する。"""
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_shipgen_kwarg("trim_mode_bow")

    def test_value_with_equals_sign(self) -> None:
        """値に = が含まれる場合、最初の = でのみ分割される。"""
        key, value = _parse_shipgen_kwarg("label=a=b")
        assert key == "label"
        assert value == "a=b"
