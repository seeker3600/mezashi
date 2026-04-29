"""Helpers for split-scoped dataset backups."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path
import shutil

from medetect.dataset_yaml import choose_splits, get_dataset_root, load_dataset_yaml

logger = logging.getLogger(__name__)

_BACKUP_ROOT_NAME = "_backup"
_VALID_KINDS = {"images", "labels"}


def backup_root(dataset_root: str | Path) -> Path:
    """Return the common backup root under one dataset root."""
    return Path(dataset_root).resolve() / _BACKUP_ROOT_NAME


def split_backup_dir(dataset_root: str | Path, kind: str, split: str) -> Path:
    """Return the backup directory for one split/kind pair."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unsupported backup kind: {kind}")
    return backup_root(dataset_root) / kind / split


def active_split_dir(dataset_root: str | Path, kind: str, split: str) -> Path:
    """Return the active dataset directory for one split/kind pair."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    return Path(dataset_root).resolve() / kind / split


def discover_split_names(
    dataset_root: str | Path,
    *,
    kinds: Sequence[str] = ("labels",),
) -> list[str]:
    """Return immediate split directory names found under the dataset."""
    root = Path(dataset_root).resolve()
    split_names: set[str] = set()
    for kind in kinds:
        if kind not in _VALID_KINDS:
            raise ValueError(f"Unsupported dataset kind: {kind}")
        kind_root = root / kind
        if not kind_root.is_dir():
            continue
        split_names.update(
            entry.name
            for entry in kind_root.iterdir()
            if entry.is_dir()
        )
    return sorted(split_names)


def resolve_dataset_root_and_splits(
    dataset: str | Path,
    splits: Sequence[str] | None = None,
) -> tuple[Path, list[str]]:
    """Resolve a dataset root and the split names to operate on."""
    dataset_path = Path(dataset).resolve()
    if dataset_path.is_dir():
        dataset_root = dataset_path
        resolved_splits = list(splits) if splits is not None else discover_split_names(dataset_root)
    else:
        config_path, config = load_dataset_yaml(dataset_path)
        dataset_root = get_dataset_root(config, config_path)
        resolved_splits = list(splits) if splits is not None else choose_splits(config, None)

    if not resolved_splits:
        resolved_splits = discover_split_names(dataset_root)
    return dataset_root, resolved_splits


def _copy_directory(source_dir: Path, dest_dir: Path, *, overwrite: bool) -> None:
    if not source_dir.exists():
        return
    if dest_dir.exists():
        if not overwrite:
            return
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, dest_dir)


def ensure_split_backup(
    dataset_root: str | Path,
    split: str,
    *,
    with_images: bool = False,
) -> None:
    """Create the initial split backup if it does not already exist."""
    root = Path(dataset_root).resolve()
    _copy_directory(
        active_split_dir(root, "labels", split),
        split_backup_dir(root, "labels", split),
        overwrite=False,
    )
    if with_images:
        _copy_directory(
            active_split_dir(root, "images", split),
            split_backup_dir(root, "images", split),
            overwrite=False,
        )


def sync_split_backup(
    dataset_root: str | Path,
    split: str,
    *,
    with_images: bool = False,
) -> None:
    """Overwrite one split backup with the current active split contents."""
    root = Path(dataset_root).resolve()
    _copy_directory(
        active_split_dir(root, "labels", split),
        split_backup_dir(root, "labels", split),
        overwrite=True,
    )
    if with_images:
        _copy_directory(
            active_split_dir(root, "images", split),
            split_backup_dir(root, "images", split),
            overwrite=True,
        )


def restore_split_from_backup(
    dataset_root: str | Path,
    split: str,
    *,
    with_images: bool = False,
) -> None:
    """Restore one split from backup by copying it back to the active tree."""
    root = Path(dataset_root).resolve()
    for kind in ("labels", "images") if with_images else ("labels",):
        backup_dir = split_backup_dir(root, kind, split)
        if not backup_dir.is_dir():
            raise FileNotFoundError(f"backup {kind} split not found: {backup_dir}")

        active_dir = active_split_dir(root, kind, split)
        if active_dir.exists():
            shutil.rmtree(active_dir)
        active_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup_dir, active_dir)


def restore_dataset_splits(
    dataset: str | Path,
    splits: Sequence[str] | None = None,
    *,
    with_images: bool = False,
) -> None:
    """Restore one or more splits from backup into the active dataset tree."""
    dataset_root, resolved_splits = resolve_dataset_root_and_splits(dataset, splits)
    for split in resolved_splits:
        restore_split_from_backup(dataset_root, split, with_images=with_images)
        logger.info(
            "Restored split '%s' from %s",
            split,
            backup_root(dataset_root),
        )
