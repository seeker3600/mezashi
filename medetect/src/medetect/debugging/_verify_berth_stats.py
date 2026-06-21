"""Temporary verification script — DELETE AFTER USE.

Instruments datagen internals to verify the bias analysis:
1. _berth_segment_frame rejection rate
2. berth_runs empty rate
3. _place_alongside_berthed_run failure reasons
4. 4-quadrant event distribution

Run:
    pixi run python -m medetect.debugging._verify_berth_stats
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

_counters: Counter[str] = Counter()


def _install_patches() -> dict[str, Any]:
    import medetect.datagen.placement as P
    import medetect.datagen.compose as C

    originals: dict[str, Any] = {}

    # ── _berth_segment_frame ────────────────────────────────────────────────
    _orig_bsf = P._berth_segment_frame  # type: ignore[attr-defined]
    originals["_berth_segment_frame"] = _orig_bsf

    def _patched_bsf(segment, water_mask):
        import math
        (x0, y0), (x1, y1) = segment
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            _counters["bsf_degenerate"] += 1
            return None
        tx, ty = dx / seg_len, dy / seg_len
        mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sample_dist = 2.0
        candidates = [(-ty, tx), (ty, -tx)]
        any_water = False
        best_score = -1
        for nx, ny in candidates:
            water_score = 0
            land_score = 0
            for offset in (sample_dist, sample_dist * 2.0, sample_dist * 3.0):
                if P._mask_contains(water_mask, mid_x + nx * offset, mid_y + ny * offset):  # type: ignore[attr-defined]
                    water_score += 1
                if P._mask_contains(water_mask, mid_x - nx * offset, mid_y - ny * offset):  # type: ignore[attr-defined]
                    land_score += 1
            score = water_score - land_score
            if water_score > 0:
                any_water = True
            if water_score > 0 and score > best_score:
                best_score = score
        result = _orig_bsf(segment, water_mask)
        if result is None:
            if not any_water:
                _counters["bsf_reject_no_water"] += 1
            else:
                _counters["bsf_reject_bad_score"] += 1
        else:
            _counters["bsf_pass"] += 1
        return result

    P._berth_segment_frame = _patched_bsf  # type: ignore[attr-defined]

    # ── _build_berth_runs ───────────────────────────────────────────────────
    _orig_bbr = P._build_berth_runs  # type: ignore[attr-defined]
    originals["_build_berth_runs"] = _orig_bbr

    def _patched_bbr(berth_segments, water_mask):
        result = _orig_bbr(berth_segments, water_mask)
        _counters["bbr_calls"] += 1
        _counters["bbr_segments_in"] += len(berth_segments)
        _counters["bbr_runs_out"] += len(result)
        if not result:
            _counters["bbr_empty"] += 1
        return result

    P._build_berth_runs = _patched_bbr  # type: ignore[attr-defined]
    C._build_berth_runs = _patched_bbr  # type: ignore[attr-defined]

    # ── _place_alongside_berthed_run ────────────────────────────────────────
    _orig_pal = P._place_alongside_berthed_run  # type: ignore[attr-defined]
    originals["_place_alongside_berthed_run"] = _orig_pal

    def _patched_pal(run, mid_sample, berth_water_mask, occupancy, svg_metas,
                     resolution_m, rng, n_ships, class_id, image_size,
                     length_range, length_exponent, size_thresholds, mixed,
                     offnadir_deg, sensor_az_world_deg, shipgen_kwargs=None):
        _counters["pal_calls"] += 1
        result = _orig_pal(run, mid_sample, berth_water_mask, occupancy, svg_metas,
                           resolution_m, rng, n_ships, class_id, image_size,
                           length_range, length_exponent, size_thresholds, mixed,
                           offnadir_deg, sensor_az_world_deg, shipgen_kwargs=shipgen_kwargs)
        if result is None:
            _counters["pal_fail"] += 1
        else:
            placed, _ = result
            _counters["pal_success"] += 1
            _counters[f"pal_success_ships_{len(placed)}"] += 1
        return result

    P._place_alongside_berthed_run = _patched_pal  # type: ignore[attr-defined]

    # ── _place_berthed_cluster ──────────────────────────────────────────────
    _orig_pbc = P._place_berthed_cluster  # type: ignore[attr-defined]
    originals["_place_berthed_cluster"] = _orig_pbc

    def _patched_pbc(berth_water_mask, occupancy, berth_segments, svg_metas,
                     resolution_m, rng, n_ships, alpha_range, class_id, image_size,
                     background, length_range, length_exponent, size_thresholds, mixed,
                     berth_stern, **kwargs):
        _counters["pbc_calls"] += 1
        _counters[f"pbc_berth_stern_{berth_stern}"] += 1
        _counters[f"pbc_n_ships_requested_{n_ships}"] += 1
        result = _orig_pbc(berth_water_mask, occupancy, berth_segments, svg_metas,
                           resolution_m, rng, n_ships, alpha_range, class_id, image_size,
                           background, length_range, length_exponent, size_thresholds, mixed,
                           berth_stern, **kwargs)
        if not result:
            _counters["pbc_fail"] += 1
        else:
            _counters["pbc_success"] += 1
            _counters[f"pbc_labels_{len(result)}"] += 1
        return result

    P._place_berthed_cluster = _patched_pbc  # type: ignore[attr-defined]
    C._place_berthed_cluster = _patched_pbc  # type: ignore[attr-defined]

    # ── _place_cluster ──────────────────────────────────────────────────────
    _orig_pc = P._place_cluster  # type: ignore[attr-defined]
    originals["_place_cluster"] = _orig_pc

    def _patched_pc(water_mask, occupancy, svg_metas, resolution_m, rng,
                    cluster_size_range, blur_sigma, alpha_range, class_id,
                    image_size, background, length_range=None, length_exponent=1.0,
                    size_thresholds=None, mixed_prob=0.5,
                    berth_prob=0.25, berth_stern_prob=0.5, berth_water_mask=None,
                    berth_segments=None, berth_runs=None, **kwargs):
        _counters["pc_calls"] += 1
        if berth_water_mask is not None and berth_segments:
            _counters["pc_berth_available"] += 1
        else:
            _counters["pc_no_berth_data"] += 1
        result = _orig_pc(water_mask, occupancy, svg_metas, resolution_m, rng,
                          cluster_size_range, blur_sigma, alpha_range, class_id,
                          image_size, background,
                          length_range=length_range, length_exponent=length_exponent,
                          size_thresholds=size_thresholds, mixed_prob=mixed_prob,
                          berth_prob=berth_prob, berth_stern_prob=berth_stern_prob,
                          berth_water_mask=berth_water_mask,
                          berth_segments=berth_segments,
                          berth_runs=berth_runs, **kwargs)
        if result:
            _counters["pc_success"] += 1
        else:
            _counters["pc_fail"] += 1
        return result

    P._place_cluster = _patched_pc  # type: ignore[attr-defined]
    C._place_cluster = _patched_pc  # type: ignore[attr-defined]

    # ── per-tile tracker: patch _compose_one to count tiles ──────────────────
    _orig_ct = C._compose_one  # type: ignore[attr-defined]
    originals["_compose_one"] = _orig_ct

    def _patched_ct(*args, **kwargs):
        result = _orig_ct(*args, **kwargs)
        _counters["tile_total"] += 1
        return result

    C._compose_one = _patched_ct  # type: ignore[attr-defined]

    return originals


def _restore_patches(originals: dict[str, Any]) -> None:
    import medetect.datagen.placement as P
    import medetect.datagen.compose as C
    for name, func in originals.items():
        if hasattr(P, name):
            setattr(P, name, func)
    C._build_berth_runs = originals["_build_berth_runs"]  # type: ignore[attr-defined]
    C._place_cluster = originals["_place_cluster"]  # type: ignore[attr-defined]
    C._place_berthed_cluster = originals["_place_berthed_cluster"]  # type: ignore[attr-defined]
    C._compose_one = originals["_compose_one"]  # type: ignore[attr-defined]


def _print_report() -> None:
    print("\n" + "=" * 70)
    print("BERTH PLACEMENT VERIFICATION REPORT")
    print("=" * 70)

    print("\n--- Tiles ---")
    print(f"  Total tiles processed    : {_counters['tile_total']}")

    print("\n--- _berth_segment_frame (per segment) ---")
    bsf_pass = _counters["bsf_pass"]
    bsf_rej_nw = _counters["bsf_reject_no_water"]
    bsf_rej_bs = _counters["bsf_reject_bad_score"]
    bsf_degen = _counters["bsf_degenerate"]
    bsf_total = bsf_pass + bsf_rej_nw + bsf_rej_bs + bsf_degen
    print(f"  Total segments evaluated : {bsf_total}")
    print(f"  Pass                     : {bsf_pass}  ({100*bsf_pass/max(1,bsf_total):.1f}%)")
    print(f"  Reject (no water nearby) : {bsf_rej_nw}  ({100*bsf_rej_nw/max(1,bsf_total):.1f}%)")
    print(f"  Reject (bad score)       : {bsf_rej_bs}  ({100*bsf_rej_bs/max(1,bsf_total):.1f}%)")
    print(f"  Degenerate (<1e-6 len)   : {bsf_degen}")

    print("\n--- _build_berth_runs (per tile with coastline) ---")
    bbr_calls = _counters["bbr_calls"]
    bbr_empty = _counters["bbr_empty"]
    print(f"  Calls                    : {bbr_calls}")
    print(f"  Segments in (total)      : {_counters['bbr_segments_in']}")
    print(f"  Runs out (total)         : {_counters['bbr_runs_out']}")
    print(f"  Empty runs (tile fails)  : {bbr_empty}  ({100*bbr_empty/max(1,bbr_calls):.1f}%)")
    avg_segs = _counters["bbr_segments_in"] / max(1, bbr_calls)
    print(f"  Avg segments per tile    : {avg_segs:.1f}")

    print("\n--- _place_berthed_cluster calls ---")
    pbc = _counters["pbc_calls"]
    print(f"  Total calls              : {pbc}")
    print(f"  berth_stern=True         : {_counters['pbc_berth_stern_True']}")
    print(f"  berth_stern=False        : {_counters['pbc_berth_stern_False']}")
    print(f"  Success                  : {_counters['pbc_success']}  ({100*_counters['pbc_success']/max(1,pbc):.1f}%)")
    print(f"  Fail (empty runs)        : {_counters['pbc_fail']}  ({100*_counters['pbc_fail']/max(1,pbc):.1f}%)")
    for k, v in sorted(_counters.items()):
        if k.startswith("pbc_labels_"):
            print(f"    labels={k.split('_')[-1]}: {v}")

    print("\n--- _place_alongside_berthed_run ---")
    pal = _counters["pal_calls"]
    print(f"  Total calls              : {pal}")
    print(f"  Success                  : {_counters['pal_success']}  ({100*_counters['pal_success']/max(1,pal):.1f}%)")
    print(f"  Fail                     : {_counters['pal_fail']}  ({100*_counters['pal_fail']/max(1,pal):.1f}%)")
    for k, v in sorted(_counters.items()):
        if k.startswith("pal_success_ships_"):
            print(f"    placed={k.split('_')[-1]}: {v}")

    print("\n--- _place_cluster (cluster events) ---")
    pc = _counters["pc_calls"]
    print(f"  Total calls              : {pc}")
    print(f"  berth data available     : {_counters['pc_berth_available']}")
    print(f"  no berth data            : {_counters['pc_no_berth_data']}")
    print(f"  Success (any labels)     : {_counters['pc_success']}")
    print(f"  Fail (no labels)         : {_counters['pc_fail']}")

    print("\n--- n_ships requested to _place_berthed_cluster ---")
    for k, v in sorted(_counters.items()):
        if k.startswith("pbc_n_ships_requested_"):
            n = k.split("_")[-1]
            print(f"  n={n}: {v}")

    print("=" * 70)


def main() -> None:
    here = Path(__file__).resolve().parent.parent.parent.parent
    bg_dir = here / "datasets" / "sentinel2_visual"
    coastline = here / "datasets" / "coastlines-split-4326" / "lines.shp"

    if not bg_dir.exists():
        print(f"ERROR: bg_dir not found: {bg_dir}")
        sys.exit(1)
    if not coastline.exists():
        print(f"ERROR: coastline not found: {coastline}")
        sys.exit(1)

    print("Installing patches...")
    originals = _install_patches()

    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Running datagen (20 images) into {tmpdir} ...")
            from medetect.datagen import generate_dataset
            generate_dataset(
                bg_dir=bg_dir,
                output_dir=Path(tmpdir),
                count=20,
                image_size=640,
                geo_scale=0.5,
                ships_per_image=(1, 60),
                cluster_prob=0.8,
                cluster_size=(2, 10),
                resolution=1.0,
                ship_length_range=(30.0, 80.0),
                length_exponent=3.0,
                ship_alpha=(1.0, 1.0),
                berth_prob=1.0,
                berth_stern_prob=0.5,
                coastline=coastline,
                seed=42,
                max_workers=0,
                override=True,
            )
    finally:
        print("Restoring patches...")
        _restore_patches(originals)

    _print_report()


if __name__ == "__main__":
    main()
