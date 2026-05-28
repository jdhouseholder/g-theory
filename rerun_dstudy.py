from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import gtheory
from sweeps.common import compute_het_artifacts


# WildBench has 1023 items, so its recommendation n_I goes to 1000; the default
# gtheory.DSTUDY_NI tops out at 500.
_WILDBENCH_GRID = ([25, 50, 100, 200, 500, 1000], [1, 2, 3, 5, 8])

SWEEPS = [
    (
        "sweep_helm",
        "per_cell",
        ("scenario", "criterion"),
        lambda name: tuple(name.split("__", 1)),
        "aggregate_dstudy.csv",
        False,
        None,
    ),
    (
        "sweep_helm_het",
        "per_cell",
        ("scenario", "criterion"),
        lambda name: tuple(name.split("__", 1)),
        "aggregate_dstudy_het.csv",
        True,
        None,
    ),
    (
        "sweep_biggen",
        "per_capability",
        ("capability",),
        lambda name: (name,),
        "aggregate_dstudy.csv",
        False,
        None,
    ),
    (
        "sweep_biggen_het",
        "per_capability",
        ("capability",),
        lambda name: (name,),
        "aggregate_dstudy_het.csv",
        True,
        None,
    ),
    (
        "sweep_wildbench",
        "per_cap",
        ("cap",),
        lambda name: (name.removeprefix("n_M_"),),
        "aggregate_dstudy.csv",
        False,
        _WILDBENCH_GRID,
    ),
    (
        "sweep_wildbench_het",
        "per_cap",
        ("cap",),
        lambda name: (name.removeprefix("n_M_"),),
        "aggregate_dstudy_het.csv",
        True,
        _WILDBENCH_GRID,
    ),
]


def _dstudy_for_cell(
    cell_dir: Path,
    is_het: bool,
    ni_grid,
    nj_grid,
) -> pd.DataFrame | None:
    if is_het:
        path = cell_dir / "samples_het.npz"
        if not path.exists():
            return None
        samples = gtheory.load_dstudy_samples(path)
        judges = (
            samples["judges"].tolist()
            if "judges" in samples
            else [f"j{i}" for i in range(samples["s_j"].shape[1])]
        )
        het = compute_het_artifacts(samples, judges, ni_grid, nj_grid)
        return het["ds_het"]

    lmm_path = cell_dir / "samples_lmm.npz"
    glmm_path = cell_dir / "samples_glmm.npz"
    if not lmm_path.exists() or not glmm_path.exists():
        return None
    lmm = gtheory.load_dstudy_samples(lmm_path)
    glmm = gtheory.load_dstudy_samples(glmm_path)
    ds_lmm = gtheory.latent_dstudy(
        lmm, residual_var=None, ni_grid=ni_grid, nj_grid=nj_grid
    )
    ds_lmm.insert(0, "model", "LMM")
    ds_glmm = gtheory.latent_dstudy(
        glmm,
        residual_var=gtheory.LOGIT_RESIDUAL_VAR,
        ni_grid=ni_grid,
        nj_grid=nj_grid,
    )
    ds_glmm.insert(0, "model", "GLMM")
    return pd.concat([ds_lmm, ds_glmm], ignore_index=True)


def _rebuild_sweep(
    sweep_root: Path,
    per_cell_name: str,
    group_cols: tuple[str, ...],
    name_to_group,
    aggregate_csv: str,
    is_het: bool,
    ni_grid,
    nj_grid,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    per_cell_root = sweep_root / per_cell_name
    if not per_cell_root.exists():
        return 0, 0

    cell_dirs = sorted(p for p in per_cell_root.iterdir() if p.is_dir())
    pieces = []
    n_hit = 0
    for cell_dir in cell_dirs:
        df = _dstudy_for_cell(cell_dir, is_het, ni_grid, nj_grid)
        if df is None:
            continue
        n_hit += 1
        groups = name_to_group(cell_dir.name)
        for i, (col, val) in enumerate(zip(group_cols, groups)):
            df.insert(i, col, val)
        # Keep per-cell file in sync with the rebuilt aggregate.
        per_cell_csv = (
            "dstudy_latent_het.csv" if is_het else "dstudy_latent.csv"
        )
        if not dry_run:
            df.drop(columns=list(group_cols)).to_csv(
                cell_dir / per_cell_csv, index=False
            )
        pieces.append(df)

    n_total = len(cell_dirs)
    if not pieces:
        print(f"  {sweep_root.name}: 0/{n_total} cells have cached samples - skip")
        return 0, n_total

    out = pd.concat(pieces, ignore_index=True)
    out_path = sweep_root / aggregate_csv
    if dry_run:
        print(
            f"  {sweep_root.name}: would rewrite {out_path.name} "
            f"({n_hit}/{n_total} cells, {len(out)} rows)"
        )
    else:
        out.to_csv(out_path, index=False)
        print(
            f"  {sweep_root.name}: rewrote {out_path.name} "
            f"({n_hit}/{n_total} cells, {len(out)} rows)"
        )
    return n_hit, n_total


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild aggregate_dstudy CSVs from cached posterior samples."
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("state"),
        help="Root containing sweep_helm, sweep_biggen, ...",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to specific sweep names (e.g. sweep_helm sweep_biggen_het).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be rebuilt without writing files.",
    )
    args = parser.parse_args()

    print(
        f"[rerun_dstudy] default grid: n_I={gtheory.DSTUDY_NI}  n_J={gtheory.DSTUDY_NJ}"
    )
    print(f"[rerun_dstudy] state dir: {args.state_dir.resolve()}")
    n_with = 0
    n_seen = 0
    for entry in SWEEPS:
        (sweep_name, per_cell, group_cols, name_fn, agg_csv, is_het, grid) = entry
        if args.only and sweep_name not in args.only:
            continue
        sweep_root = args.state_dir / sweep_name
        if not sweep_root.exists():
            print(f"  {sweep_name}: missing dir - skip")
            continue
        ni_grid, nj_grid = grid if grid is not None else (
            gtheory.DSTUDY_NI,
            gtheory.DSTUDY_NJ,
        )
        if grid is not None:
            print(f"  {sweep_name}: grid override n_I={ni_grid}  n_J={nj_grid}")
        hit, total = _rebuild_sweep(
            sweep_root,
            per_cell,
            group_cols,
            name_fn,
            agg_csv,
            is_het,
            ni_grid,
            nj_grid,
            dry_run=args.dry_run,
        )
        n_with += hit
        n_seen += total

    print(
        f"\n[rerun_dstudy] {n_with}/{n_seen} cells rebuilt. "
        "Cells without samples_*.npz need a NUTS rerun."
    )
    if not args.dry_run and n_with > 0:
        print(
            "[rerun_dstudy] Next: rerun `python dstudy_recommendations.py` to "
            "refresh the wide / cilow recommendation tables."
        )


if __name__ == "__main__":
    main()
