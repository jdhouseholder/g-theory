from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtheory
from loaders.wildbench import MODELS_INTERSECTION, fetch_all, load_wildbench
from sweeps.common import (
    Cell,
    DatasetSpec,
    add_common_args,
    fit_het,
    run_het_sweep,
)
from sweeps.sweep_wildbench import model_subset

DSTUDY_NI = [25, 50, 100, 200, 500, 1000]
DSTUDY_NJ = [1, 2, 3, 5, 8]

CAPS = [4, 8, 16, 41]


def _build_cells(only: list[int] | None, limit: int | None) -> list[Cell]:
    caps = CAPS if not only else only
    if limit:
        caps = caps[:limit]
    return [
        Cell(
            key={"cap": cap},
            dirname=f"n_M_{cap}",
            seed_label=f"wb-het-{cap}",
            payload={"models": model_subset(cap)},
        )
        for cap in caps
    ]


def _fit_cell(cell, **kw):
    models = cell.payload["models"]
    df = load_wildbench(models)
    print(f"  cap={cell.key['cap']}  models={models}")
    return fit_het(
        df,
        label_prefix=f"wb-cap{cell.key['cap']}",
        seed_label=cell.seed_label,
        var_filename="variance_components.csv",
        **kw,
    )


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--only",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to specific caps (e.g. --only 4 8).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prior-mj",
        type=float,
        default=None,
        help="Override config.prior_sigma['MJ'] (default 1.0). "
        "Use 0.5 to break M<->MxJ identifiability at large n_M.",
    )
    parser.add_argument("--out-dir", default="./state/sweep_wildbench_het")
    args = parser.parse_args()

    config = gtheory.default_config().with_overrides(n_categories=10)
    if args.prior_mj is not None:
        config = config.with_overrides(prior_sigma={"MJ": float(args.prior_mj)})
        print(f"[sweep_wildbench_het] config.prior_sigma['MJ'] = {args.prior_mj}")

    print(
        f"[sweep_wildbench_het] n_categories={config.n_categories} "
        f"n_cutpoints={config.n_cutpoints}"
    )

    fetch_all(verbose=False)

    spec = DatasetSpec(
        name="sweep_wildbench_het",
        cells=_build_cells(args.only, args.limit),
        fit_cell=_fit_cell,
        config=config,
        cell_subdir="per_cap",
        ni_grid=DSTUDY_NI,
        nj_grid=DSTUDY_NJ,
    )

    run_het_sweep(spec, args)


if __name__ == "__main__":
    main()
