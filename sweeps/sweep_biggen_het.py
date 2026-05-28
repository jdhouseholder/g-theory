from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtheory
from loaders.biggen import CAPABILITIES, load_biggen
from sweeps.common import (
    Cell,
    DatasetSpec,
    add_common_args,
    fit_het,
    run_het_sweep,
)


def _build_cells(only: list[str] | None, limit: int | None) -> list[Cell]:
    caps = CAPABILITIES if not only else only
    if limit:
        caps = caps[:limit]
    return [
        Cell(
            key={"capability": cap},
            dirname=cap,
            seed_label=f"biggen-het-{cap}",
        )
        for cap in caps
    ]


def _fit_cell(cell, **kw):
    df = load_biggen(cell.key["capability"])
    return fit_het(
        df,
        label_prefix=cell.key["capability"],
        seed_label=cell.seed_label,
        **kw,
    )


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--alpha-centered",
        type=float,
        default=1.0,
        help="Partial centering for the M effect (config.reparam['alpha']). "
        "Default 1.0 (centered) - matches sweep_biggen.py.",
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="./state/sweep_biggen_het")
    args = parser.parse_args()

    config = gtheory.default_config()
    if args.alpha_centered is not None:
        config = config.with_overrides(reparam={"alpha": float(args.alpha_centered)})
        print(f"[sweep_biggen_het] config.reparam['alpha'] = {args.alpha_centered}")

    spec = DatasetSpec(
        name="sweep_biggen_het",
        cells=_build_cells(args.only, args.limit),
        fit_cell=_fit_cell,
        config=config,
        cell_subdir="per_capability",
        ni_grid=gtheory.DSTUDY_NI,
        nj_grid=gtheory.DSTUDY_NJ,
    )

    run_het_sweep(spec, args)


if __name__ == "__main__":
    main()
