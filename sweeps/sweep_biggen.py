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
    fit_vanilla,
    run_vanilla_sweep,
)

def _build_cells(only: list[str] | None, limit: int | None) -> list[Cell]:
    caps = CAPABILITIES if not only else only
    if limit:
        caps = caps[:limit]
    return [Cell(key={"capability": cap}, dirname=cap, seed_label=cap) for cap in caps]


def _fit_cell(cell, **kw):
    df = load_biggen(cell.key["capability"])
    return fit_vanilla(df, label_prefix=cell.key["capability"],
                       seed_label=cell.seed_label, **kw)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--alpha-centered",
        type=float,
        default=1.0,
        help="Partial centering for the M effect (config.reparam['alpha']). "
        "Default 1.0 (centered) - BiGGen sigma_M is large enough that NC drives "
        "NUTS to ~100%% divergence.",
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prior-mj",
        type=float,
        default=None,
        help="Override config.prior_sigma['MJ'] (default 1.0).",
    )
    parser.add_argument(
        "--prior-ij",
        type=float,
        default=None,
        help="Override config.prior_sigma['IJ'] (default 1.0).",
    )
    parser.add_argument("--out-dir", default="./state/sweep_biggen")
    args = parser.parse_args()

    config = gtheory.default_config()
    if args.alpha_centered is not None:
        config = config.with_overrides(reparam={"alpha": float(args.alpha_centered)})
        print(f"[sweep_biggen] config.reparam['alpha'] = {args.alpha_centered}")
    prior_overrides = {}
    if args.prior_mj is not None:
        prior_overrides["MJ"] = float(args.prior_mj)
    if args.prior_ij is not None:
        prior_overrides["IJ"] = float(args.prior_ij)
    if prior_overrides:
        config = config.with_overrides(prior_sigma=prior_overrides)
        print(f"[sweep_biggen] prior_sigma overrides = {prior_overrides}")

    spec = DatasetSpec(
        name="sweep_biggen",
        cells=_build_cells(args.only, args.limit),
        fit_cell=_fit_cell,
        config=config,
        cell_subdir="per_capability",
        ni_grid=gtheory.DSTUDY_NI,
        nj_grid=gtheory.DSTUDY_NJ,
    )

    run_vanilla_sweep(spec, args)


if __name__ == "__main__":
    main()
