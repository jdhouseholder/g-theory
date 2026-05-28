from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtheory
from loaders.wildbench import MODELS_INTERSECTION, fetch_all, load_wildbench
from sweeps.common import (
    Cell,
    DatasetSpec,
    add_common_args,
    fit_vanilla,
    run_vanilla_sweep,
)

DSTUDY_NI = [25, 50, 100, 200, 500, 1000]
DSTUDY_NJ = [1, 2, 3, 5, 8]

CAPS = [4, 8, 16, 41]


def model_subset(cap: int, seed: int = gtheory.SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(MODELS_INTERSECTION)
    return list(perm[:cap])


def _build_cells(only: list[int] | None) -> list[Cell]:
    caps = CAPS if not only else only
    return [
        Cell(
            key={"cap": cap},
            dirname=f"n_M_{cap}",
            seed_label=f"wb-{cap}",
            payload={"models": model_subset(cap)},
        )
        for cap in caps
    ]


def _fit_cell(cell, **kw):
    models = cell.payload["models"]
    df = load_wildbench(models)
    print(f"  cap={cell.key['cap']}  models={models}")
    return fit_vanilla(df, label_prefix=f"wb-cap{cell.key['cap']}",
                       seed_label=cell.seed_label, **kw)


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
    parser.add_argument("--out-dir", default="./state/sweep_wildbench")
    args = parser.parse_args()

    # gtheory's default ModelConfig uses N_CATEGORIES=5; WildBench is 1-10.
    config = gtheory.default_config().with_overrides(n_categories=10)
    print(
        f"[sweep_wildbench] n_categories={config.n_categories} "
        f"n_cutpoints={config.n_cutpoints}"
    )

    fetch_all(verbose=False)

    spec = DatasetSpec(
        name="sweep_wildbench",
        cells=_build_cells(args.only),
        fit_cell=_fit_cell,
        config=config,
        cell_subdir="per_cap",
        ni_grid=DSTUDY_NI,
        nj_grid=DSTUDY_NJ,
    )

    run_vanilla_sweep(spec, args)


if __name__ == "__main__":
    main()
