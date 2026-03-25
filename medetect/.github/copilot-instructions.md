# medetect — AI agent instructions

## Project overview
Python package for satellite/aerial image object-detection model training & export.  
Build system: **Hatch** / Runtime & dependency management: **pixi** / Tests: **pytest**

## Tooling — prefer dedicated tools over shell

| Task | Preferred approach | Shell fallback |
|---|---|---|
| Run tests | VS Code *Run Tests* (pytest) | `pixi run pytest` |
| Run single test | VS Code *Run Tests* with filter | `pixi run pytest tests/xview/test_slice.py -k test_name` |
| Install deps | — | `pixi install` |
| Add dep (conda) | — | `pixi add <pkg>` |
| Add dep (PyPI) | — | `pixi add --pypi <pkg>` |
| Lint / format | VS Code formatter (Ruff / Black) | — |
| Run a task | — | `pixi run <task>` |

> When running from the monorepo root, use `pixi -m medetect/pyproject.toml ...`.

## Test-first development
1. **Write a failing test first** in `tests/` mirroring the `src/medetect/` structure.
2. Implement the minimum code to make it pass.
3. Refactor if needed, keeping tests green.

### Testing conventions
- Framework: **pytest** (`testpaths = ["tests"]`).
- Group related tests in a class (`class TestXxx:`).
- Use `pytest.approx` for float comparison, `tmp_path` / `monkeypatch` for isolation.
- Docstrings in Japanese describing the test purpose.
- Test pure logic first; avoid heavy I/O or GPU in unit tests.

```python
# Example: tests/xview/test_slice.py
class TestComputeGeoResolution:
    def test_projected_returns_average(self) -> None:
        """投影座標系ではそのまま平均を返す。"""
        result = _compute_geo_resolution(2.0, 4.0, is_geographic=False)
        assert result == pytest.approx(3.0)
```

## Coding rules
- Keep changes **minimal and focused**. Do not touch files unrelated to the task.
- No cosmetic-only edits (whitespace, blank lines, import reordering) in untouched code.
- Prefer **reasonable DRY** over perfect backward compatibility.
- `any` and `type: ignore` require a justifying comment.
- 大胆に変更せよ。上記と矛盾するように見えるかもしれないが、必要なら作り直しを恐れるな。

## Language policy
- Issue / PR text → **日本語**
- Code, comments, commit messages → English (OK)

## AI agent best practices
- **Read before edit.** Always read the target file and surrounding context before proposing changes.
- **Scope check.** If the request touches model I/O or label maps, verify whether `mefront/` side also needs updating — but do NOT silently edit mefront files.
- **No speculation.** If unclear, search the codebase or ask rather than guessing API signatures.
- **No new files unless necessary.** Prefer editing existing files over creating new ones.
- **Verify after edit.** Run the relevant tests (or at minimum check for lint/type errors) after making changes.
- **One concern per commit.** Do not bundle unrelated fixes.
