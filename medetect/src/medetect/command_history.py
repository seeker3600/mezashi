from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
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
) -> Path:
    root = Path(dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp": _timestamp_now(),
        "command": command,
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd().resolve() if cwd is None else Path(cwd).resolve()),
        "dataset_root": str(root),
        "status": status,
    }
    if result is not None:
        record["result"] = result

    log_path = command_history_path(root)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, default=_json_default))
        handle.write("\n")
    return log_path


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)