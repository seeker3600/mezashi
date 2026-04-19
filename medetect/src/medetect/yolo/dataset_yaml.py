"""Helpers for resolving YOLO dataset YAML files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


def load_dataset_yaml(
    config_path: str | Path,
) -> tuple[Path, dict[str, object]]:
    """Load a dataset YAML file and return its resolved path plus mapping."""
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError("Dataset YAML must contain a top-level mapping.")
    return config_file, config


def resolve_dataset_root(path_value: object, config_path: Path) -> Path:
    """Resolve the dataset root specified by a YAML ``path`` entry."""
    if not isinstance(path_value, (str, Path)):
        raise TypeError("'path' must be a string or path-like value.")

    dataset_path = Path(path_value)
    if dataset_path.is_absolute():
        return dataset_path.resolve()

    config_relative = (config_path.parent / dataset_path).resolve()
    if config_relative.exists():
        return config_relative

    try:
        from ultralytics import settings
    except ImportError:
        return config_relative

    datasets_root = Path(settings["datasets_dir"]).expanduser()
    ultralytics_relative = (datasets_root / dataset_path).resolve()
    if ultralytics_relative.exists():
        return ultralytics_relative

    return ultralytics_relative


def get_dataset_root(
    config: Mapping[str, object],
    config_path: Path,
    *,
    default_to_parent: bool = False,
) -> Path:
    """Return the absolute dataset root for one dataset YAML mapping."""
    if "path" not in config:
        if default_to_parent:
            return config_path.parent.resolve()
        raise KeyError("Dataset YAML must define 'path'.")
    return resolve_dataset_root(config["path"], config_path)


def choose_splits(
    config: Mapping[str, object],
    split: str | None,
) -> list[str]:
    """Return the split keys to inspect from a dataset YAML mapping."""
    candidates = [split] if split is not None else ["train", "val", "test"]
    return [candidate for candidate in candidates if candidate in config]


def resolve_split_dirs(
    split_value: object,
    root: Path,
) -> list[Path]:
    """Resolve one or more split path values to absolute directories."""
    if isinstance(split_value, (str, Path)):
        split_values = [split_value]
    elif isinstance(split_value, list):
        split_values = split_value
    else:
        raise TypeError("Split value must be a path or list of paths.")

    dirs: list[Path] = []
    for value in split_values:
        if not isinstance(value, (str, Path)):
            raise TypeError("Each split path must be a string or path-like value.")
        directory = Path(value)
        if not directory.is_absolute():
            directory = root / directory
        dirs.append(directory.resolve())
    return dirs