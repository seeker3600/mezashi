# Repository overview (monorepo)
- medetect/: Python (training/export)
- mefront/: Node.js (browser frontend/inference)

# General rules
- Prefer minimal, consistent changes. Do not mix concerns between medetect/ and mefront/ unless required.
- Before suggesting commands, check each subproject's README and config files (pyproject.toml/package.json).
- Keep generated artifacts out of Git unless explicitly tracked; document how to fetch/build them.
- When changing model I/O or label map, update shared/ metadata and any consuming code.
