from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtheory
from loaders.helm import all_cells, load_helm
from sweeps.common import (
    Cell,
    DatasetSpec,
    add_common_args,
    fit_vanilla,
    run_vanilla_sweep,
)


def _build_cells(only: list[str] | None, limit: int | None) -> list[Cell]:
    cells = all_cells()
    if only:
        cells = [(s, c) for (s, c) in cells if s in set(only)]
    if limit:
        cells = cells[:limit]
    return [
        Cell(key={"scenario": s, "criterion": c}, dirname=f"{s}__{c}")
        for (s, c) in cells
    ]


def _write_helm_extras(df: pd.DataFrame, enc, scenario: str, criterion: str, out_dir) -> None:
    df.to_csv(out_dir / "long_form_data.csv", index=False)
    (out_dir / "data_summary.txt").write_text(
        f"scenario  : {scenario}\n"
        f"criterion : {criterion}\n"
        f"n_models  : {enc.n_models} {enc.models}\n"
        f"n_items   : {enc.n_items}\n"
        f"n_judges  : {enc.n_judges} {enc.judges}\n"
        f"n_obs     : {len(df)}\n"
        f"score_ord : {dict(sorted(df.score_ord.value_counts().items()))}\n"
    )


def _write_helm_recommendations(res: dict, target: float, out_dir) -> pd.DataFrame:
    ds_latent = res["ds_latent"]
    ds_prob = res.get("ds_prob", pd.DataFrame())
    sources = [
        ("LMM_latent", ds_latent[ds_latent.model == "LMM"]),
        ("GLMM_latent", ds_latent[ds_latent.model == "GLMM"]),
    ]
    if not ds_prob.empty:
        sources += [
            ("LMM_probability", ds_prob[ds_prob.model == "LMM"]),
            ("GLMM_probability", ds_prob[ds_prob.model == "GLMM"]),
        ]
    rec_rows = []
    for src_label, src_df in sources:
        for use_low in (False, True):
            rec = gtheory.cheapest_design(src_df, target=target, use_lower_ci=use_low)
            rec["source"] = src_label
            rec["criterion_kind"] = "CI lower" if use_low else "median"
            rec_rows.append(rec)
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(out_dir / "recommendations.csv", index=False)
    return rec_df


def _write_helm_summary(df, enc, res, scenario, criterion, out_dir) -> None:
    m = res["meta"]
    div_note = f"divergences: LMM={m['div_lmm']}, GLMM={m['div_glmm']}"
    lines = [
        "gtheory: HELM Instruct G-theory + D-study (NumPyro / NUTS)",
        "=" * 60,
        f"scenario  : {scenario}",
        f"criterion : {criterion}",
        f"design    : {enc.n_models} models x {enc.n_items} items x {enc.n_judges} judges  ({len(df)} obs)",
        f"NUTS time : LMM {m['t_lmm']:.1f}s,  GLMM {m['t_glmm']:.1f}s",
        div_note,
        "",
        "Variance components (posterior median, 95% CI):",
        res["var_df"].to_string(index=False),
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines))


def _make_fit_cell(args):
    def fit_cell(cell, **kw):
        scenario = cell.key["scenario"]
        criterion = cell.key["criterion"]
        df = load_helm(scenario, criterion)
        enc = gtheory.encode(df)
        out_dir = kw["out_dir"]
        _write_helm_extras(df, enc, scenario, criterion, out_dir)
        res = fit_vanilla(
            df,
            label_prefix=f"{scenario}/{criterion}",
            seed_label="",  # matches legacy single-cell HELM PRNG seeding
            skip_prob_scale=args.skip_prob_scale,
            **kw,
        )
        res["rec_df"] = _write_helm_recommendations(res, args.target, out_dir)
        _write_helm_summary(df, enc, res, scenario, criterion, out_dir)
        return res

    return fit_cell


def main():
    parser = argparse.ArgumentParser(description="Sweep gtheory across HELM cells.")
    add_common_args(parser, default_max_tree_depth=10)
    parser.add_argument("--target", type=float, default=0.8)
    parser.add_argument("--skip-prob-scale", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="./state/sweep_helm")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--alpha-centered",
        type=float,
        default=None,
        help="Override partial centering for the M effect (config.reparam['alpha']).",
    )
    args = parser.parse_args()

    config = gtheory.default_config()
    if args.alpha_centered is not None:
        config = config.with_overrides(reparam={"alpha": float(args.alpha_centered)})
        print(f"[sweep_helm] config.reparam['alpha'] = {args.alpha_centered}")

    spec = DatasetSpec(
        name="sweep_helm",
        cells=_build_cells(args.only, args.limit),
        fit_cell=_make_fit_cell(args),
        config=config,
        cell_subdir="per_cell",
        ni_grid=gtheory.DSTUDY_NI,
        nj_grid=gtheory.DSTUDY_NJ,
    )

    run_vanilla_sweep(spec, args, with_recommendations=True)


if __name__ == "__main__":
    main()
