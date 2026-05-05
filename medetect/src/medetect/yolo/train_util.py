import shutil
from pathlib import Path

from medetect.command_history import command_history_path
from medetect.yolo.backup import resolve_dataset_root_and_splits


def copy_training_artifacts(trainer) -> None:  # type: ignore[no-untyped-def]  # ultralytics Trainer has no stubs
    """学習開始前に dataset 履歴ログと train.py を run ディレクトリへコピーする。

    ultralytics の on_pretrain_routine_start コールバックとして呼ばれることを想定。
    """
    run_dir = Path(trainer.save_dir)
    dataset = trainer.args.data
    dataset_root, _ = resolve_dataset_root_and_splits(dataset)
    dataset_history = command_history_path(dataset_root)
    if dataset_history.is_file():
        shutil.copy(dataset_history, run_dir / dataset_history.name)
    train_py = Path(__file__).parent / "train.py"
    if train_py.is_file():
        shutil.copy(train_py, run_dir / "train.py")
