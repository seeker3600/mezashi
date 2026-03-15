from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from medetect.yolo.__main__ import main
from medetect.yolo.train import _find_latest_checkpoint, _is_oom_error, train_yolo_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oom_error() -> RuntimeError:
    """Return a RuntimeError that looks like a CUDA OOM."""
    return RuntimeError("CUDA out of memory. Tried to allocate 200 MiB")


# ---------------------------------------------------------------------------
# _is_oom_error
# ---------------------------------------------------------------------------


def test_is_oom_error_runtime_error_with_oom_message() -> None:
    assert _is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))


def test_is_oom_error_runtime_error_without_oom_message() -> None:
    assert not _is_oom_error(RuntimeError("some other error"))


def test_is_oom_error_value_error() -> None:
    assert not _is_oom_error(ValueError("out of memory"))  # not a RuntimeError


# ---------------------------------------------------------------------------
# _find_latest_checkpoint
# ---------------------------------------------------------------------------


def test_find_latest_checkpoint_returns_none_when_no_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert _find_latest_checkpoint() is None


def test_find_latest_checkpoint_returns_none_when_no_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "runs" / "detect").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert _find_latest_checkpoint() is None


def test_find_latest_checkpoint_returns_most_recent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("train1", "train2"):
        weights = tmp_path / "runs" / "detect" / name / "weights"
        weights.mkdir(parents=True)
        (weights / "last.pt").write_text(name, encoding="utf-8")

    # Make train2/weights/last.pt newer
    import time
    time.sleep(0.01)
    (tmp_path / "runs" / "detect" / "train2" / "weights" / "last.pt").touch()

    monkeypatch.chdir(tmp_path)
    result = _find_latest_checkpoint()
    assert result is not None
    assert result.parent.parent.name == "train2"


# ---------------------------------------------------------------------------
# train_yolo_model – retry behaviour
# ---------------------------------------------------------------------------


def test_train_succeeds_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_model = MagicMock()
    with patch("medetect.yolo.train.YOLO", return_value=mock_model):
        train_yolo_model(max_retries=0)
    mock_model.train.assert_called_once()


def test_train_raises_non_oom_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_model = MagicMock()
    mock_model.train.side_effect = ValueError("not an OOM error")
    with patch("medetect.yolo.train.YOLO", return_value=mock_model):
        with pytest.raises(ValueError, match="not an OOM error"):
            train_yolo_model(max_retries=3)
    # Should not have retried
    mock_model.train.assert_called_once()


def test_train_oom_raises_when_max_retries_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_model = MagicMock()
    mock_model.train.side_effect = _oom_error()
    with patch("medetect.yolo.train.YOLO", return_value=mock_model):
        with pytest.raises(RuntimeError, match="out of memory"):
            train_yolo_model(max_retries=0)
    mock_model.train.assert_called_once()


def test_train_oom_raises_when_no_checkpoint_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    mock_model = MagicMock()
    mock_model.train.side_effect = _oom_error()
    with patch("medetect.yolo.train.YOLO", return_value=mock_model):
        with pytest.raises(RuntimeError, match="out of memory"):
            train_yolo_model(max_retries=2)
    mock_model.train.assert_called_once()


def test_train_oom_retries_from_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OOM on first attempt → load checkpoint → resume=True on second attempt."""
    # Prepare a fake last.pt
    weights = tmp_path / "runs" / "detect" / "train1" / "weights"
    weights.mkdir(parents=True)
    last_pt = weights / "last.pt"
    last_pt.write_text("fake", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resumed_model = MagicMock()
    initial_model = MagicMock()
    # First call raises OOM; resumed model succeeds
    initial_model.train.side_effect = _oom_error()

    yolo_instances = [initial_model, resumed_model]

    with patch("medetect.yolo.train.YOLO", side_effect=yolo_instances) as mock_yolo_cls:
        train_yolo_model(max_retries=1)

    # YOLO() called twice: once with "yolo26m.pt", once with the checkpoint path
    # _find_latest_checkpoint returns a path relative to cwd
    assert mock_yolo_cls.call_count == 2
    assert mock_yolo_cls.call_args_list[0] == call("yolo26m.pt")
    second_arg = Path(mock_yolo_cls.call_args_list[1].args[0])
    assert second_arg.name == "last.pt"
    assert second_arg.parent.name == "weights"
    assert second_arg.parent.parent.name == "train1"

    initial_model.train.assert_called_once()
    resumed_model.train.assert_called_once_with(resume=True)


def test_train_oom_raises_after_exhausting_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OOM on every attempt → raises after max_retries exhausted."""
    for name in ("train1", "train2"):
        weights = tmp_path / "runs" / "detect" / name / "weights"
        weights.mkdir(parents=True)
        (weights / "last.pt").write_text("fake", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    always_oom = MagicMock()
    always_oom.train.side_effect = _oom_error()

    with patch("medetect.yolo.train.YOLO", return_value=always_oom):
        with pytest.raises(RuntimeError, match="out of memory"):
            train_yolo_model(max_retries=2)

    # initial attempt + 2 retries = 3 total calls
    assert always_oom.train.call_count == 3


# ---------------------------------------------------------------------------
# CLI – --max-retries argument
# ---------------------------------------------------------------------------


def test_cli_train_default_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["python", "train"])
    with patch("medetect.yolo.__main__.train_yolo_model") as mock_train:
        main()
    mock_train.assert_called_once_with(max_retries=0)


def test_cli_train_explicit_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["python", "train", "--max-retries", "3"])
    with patch("medetect.yolo.__main__.train_yolo_model") as mock_train:
        main()
    mock_train.assert_called_once_with(max_retries=3)
