from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


def test_cli_train_calls_train_yolo_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """train サブコマンドは train_yolo_model を引数なしで呼ぶ。"""
    monkeypatch.setattr(sys, "argv", ["python", "train"])
    module = importlib.import_module("medetect.yolo.__main__")
    with patch.object(module, "train_yolo_model") as mock_train:
        module.main()
    mock_train.assert_called_once_with()
