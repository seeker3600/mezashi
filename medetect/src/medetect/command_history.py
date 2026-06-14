from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

COMMAND_HISTORY_FILENAME = "command_history.jsonl"


def command_history_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root).resolve() / COMMAND_HISTORY_FILENAME


def append_command_history(
    dataset_root: str | Path,
    *,
    command: str,
    argv: list[str] | tuple[str, ...] | None = None,
    cwd: str | Path | None = None,
    status: str = "success",
    result: Any | None = None,
    overwrite: bool = False,
) -> Path:
    root = Path(dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp": _timestamp_now(),
        "git_commit_hash": _git_commit_hash(),
        "command": command,
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd().resolve() if cwd is None else Path(cwd).resolve()),
        "dataset_root": str(root),
        "status": status,
    }
    if result is not None:
        record["result"] = result

    log_path = command_history_path(root)
    mode = "w" if overwrite else "a"
    with log_path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True, default=_json_default))
        handle.write("\n\n")
    return log_path


def read_command_history(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Read all records from the command history log for *dataset_root*."""
    log_path = command_history_path(dataset_root)
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            records.append(json.loads(chunk))
    return records


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit_hash = completed.stdout.strip()
    if len(commit_hash) == 40:
        return commit_hash
    return None


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)