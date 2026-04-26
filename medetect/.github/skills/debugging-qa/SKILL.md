---
name: debugging-qa
description: Use this when the user explicitly asks for medetect visual QA, preview rendering, artifact inspection, or reusable debugging commands under medetect.debugging. Do not use it for production shipgen or datagen defect fixes unless the user asks for debugging tooling or existing validation is insufficient.
---

# medetect debugging QA

Use this skill when the user explicitly wants visual inspection, beam-profile validation, artifact inspection, or reusable debugging commands in this repository.

Do not use this skill just because the defect is visual. When the problem is a production behavior in `src/medetect/shipgen/` or `src/medetect/datagen/`, first inspect and modify the owning module and nearby tests. Extend `src/medetect/debugging/` only if the user asked for repro tooling or the existing validation surface cannot express the needed observation.

## Core rules

1. Prefer tracked debugging modules under `src/medetect/debugging/` over ad hoc scripts in `samples/`.
1. Put generated inspection artifacts under `debug_runs/`, not `samples/`.
1. On Windows, do not run `.pixi/envs/.../python.exe` directly for NumPy/Pillow work. Use `pixi run python -m ...`.
1. For ship outline QA, validate rendered appearance after background compositing. Do not judge transparent RGB directly.
1. If a task changes ship appearance, run the ten-ship QA and inspect the generated profile images before concluding the work is done.
1. Before creating new debugging workflows, name the missing observation that existing tests, `shipgen-qa`, or `ship-preview` cannot provide.

## Main commands

Run these from the repository root.

```text
pixi run python -m medetect.debugging shipgen-qa
pixi run python -m medetect.debugging ship-preview
pixi run python -m medetect.debugging shadow-preview
pixi run python -m medetect.debugging wake-preview
pixi run python -m medetect.debugging cluster-profile <dataset-dir>
pixi run python -m medetect.debugging pixel-profile <image> <x1> <y1> <x2> <y2>
```

## Expected outputs

1. `shipgen-qa` writes `debug_runs/shipgen-profile-qa/`.
1. `ship-preview` writes `debug_runs/ship-preview/`.
1. `shadow-preview` writes `debug_runs/shadow-preview/`.
1. `wake-preview` writes `debug_runs/wake-preview/`.
1. `cluster-profile` defaults to `debug_runs/cluster-profile/cluster_profile.png`.
1. `pixel-profile` defaults to `debug_runs/pixel-profile/profile_output.png`.

## Shipgen QA interpretation

`shipgen-qa` renders a standard ten-class ship set, writes per-ship images and profile plots, and exits non-zero if a bilateral dark or bright outline is detected.

Inspect these files first:

1. `debug_runs/shipgen-profile-qa/manifest.tsv`
1. `debug_runs/shipgen-profile-qa/summary.json`
1. `debug_runs/shipgen-profile-qa/profiles/*_profile.png`

The representative row is intentionally chosen from a wide mid-body band after background compositing, using near-max-width rows and smooth-row aggregation. This avoids false alarms from superstructures and keeps automated QA aligned with manual pixel-profile checks.

## When extending debugging tools

1. Add reusable logic to `src/medetect/debugging/` only after deciding that the production fix belongs elsewhere or that a dedicated QA workflow is explicitly required.
1. Expose it through `python -m medetect.debugging ...` when it is a user-facing workflow.
1. Add focused pytest coverage for pure logic and artifact generation contracts.
1. Keep `samples/` free of active Python entry points.
