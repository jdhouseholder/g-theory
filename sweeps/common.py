from __future__ import annotations

import argparse
import json
import time
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import numpy as np
import numpyro
import pandas as pd

import gtheory


def headline_max(
    ds_long: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    sub = ds_long.copy()
    sub["budget"] = sub["n_I"] * sub["n_J"]
    sub = sub.dropna(subset=["Phi_median"])
    if sub.empty:
        return pd.DataFrame()
    by = list(group_cols) + ["model", "scale"]
    idx = sub.groupby(by)["budget"].idxmax()
    out = sub.loc[idx].reset_index(drop=True)
    cols = list(group_cols) + [
        "model",
        "scale",
        "n_I",
        "n_J",
        "budget",
        "Phi_median",
        "Phi_low",
        "Phi_high",
    ]
    return out[cols]


def lmm_vs_glmm_gap(
    headline: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    keys = list(group_cols) + ["scale", "n_I", "n_J"]
    piv = headline.pivot_table(
        index=keys,
        columns="model",
        values=["Phi_median", "Phi_low", "Phi_high"],
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["Phi_gap_median"] = piv["Phi_median_GLMM"] - piv["Phi_median_LMM"]
    return piv.sort_values(["scale", *group_cols])


def compute_het_artifacts(
    samples_het,
    judges: Sequence[str],
    ni_grid: Sequence[int],
    nj_grid: Sequence[int],
    out_dir: Path | str | None = None,
) -> dict:
    """Writes samples_het.npz to out_dir if given so rerun_dstudy.py can
    rebuild under any grid without re-running NUTS."""
    s_j = np.asarray(samples_het["s_j"])
    eff_res_per_draw = gtheory.LOGIT_RESIDUAL_VAR * (s_j**2).mean(axis=-1)
    avg_eff_res = float(np.median(eff_res_per_draw))

    s_j_summary = pd.DataFrame(
        dict(
            judge=list(judges),
            s_j_median=np.median(s_j, axis=0),
            s_j_low=np.quantile(s_j, 0.025, axis=0),
            s_j_high=np.quantile(s_j, 0.975, axis=0),
        )
    )

    ds_het = gtheory.latent_dstudy(
        samples_het, residual_var=eff_res_per_draw, ni_grid=ni_grid, nj_grid=nj_grid
    )
    ds_het.insert(0, "model", "GLMM-het")

    if out_dir is not None:
        gtheory.save_dstudy_samples(
            samples_het,
            Path(out_dir) / "samples_het.npz",
            extras={"judges": np.array(list(judges))},
        )

    return dict(
        ds_het=ds_het,
        s_j_summary=s_j_summary,
        eff_res_per_draw=eff_res_per_draw,
        avg_eff_res=avg_eff_res,
    )


@dataclass
class Cell:
    """seed_label is crc32'd to a per-cell PRNG offset; "" = no per-cell
    variation (legacy HELM single-cell seeding)."""

    key: dict[str, Any]
    dirname: str
    seed_label: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    name: str
    cells: list[Cell]
    fit_cell: Callable[..., dict]
    config: gtheory.ModelConfig
    cell_subdir: str = "per_cell"
    ni_grid: Sequence[int] | None = None
    nj_grid: Sequence[int] | None = None


def _seed_key(seed_label: str, n: int = 1):
    """n=1 returns the raw root key (no split, matches het sweeps' historical
    seeding); n>1 splits into n keys."""
    offset = zlib.crc32(seed_label.encode()) % 10000 if seed_label else 0
    root = jax.random.PRNGKey(gtheory.SEED + offset)
    if n == 1:
        return (root,)
    return jax.random.split(root, n)


def fit_vanilla(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    config: gtheory.ModelConfig,
    ni_grid: Sequence[int],
    nj_grid: Sequence[int],
    label_prefix: str,
    seed_label: str,
    warmup: int,
    samples: int,
    chains: int,
    target_accept: float,
    dense_mass: bool,
    max_tree_depth: int,
    skip_prob_scale: bool = False,
) -> dict:
    enc = gtheory.encode(df)
    print(f"  obs={len(df)}  M={enc.n_models} I={enc.n_items} J={enc.n_judges}")

    kwargs = dict(
        model_idx=enc.model_idx,
        item_idx=enc.item_idx,
        judge_idx=enc.judge_idx,
        n_models=enc.n_models,
        n_items=enc.n_items,
        n_judges=enc.n_judges,
    )
    k_lmm, k_glmm, k_sim_l, k_sim_g = _seed_key(seed_label, n=4)

    t_lmm0 = time.time()
    samples_lmm, _, div_lmm, _ = gtheory.run_nuts(
        gtheory.lmm_model,
        k_lmm,
        model_kwargs={**kwargs, "score": enc.score_raw, "config": config},
        label=f"LMM/{label_prefix}",
        warmup=warmup,
        samples=samples,
        chains=chains,
        target_accept=target_accept,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
    )
    t_lmm = time.time() - t_lmm0

    t_glmm0 = time.time()
    samples_glmm, _, div_glmm, _ = gtheory.run_nuts(
        gtheory.glmm_ordinal,
        k_glmm,
        model_kwargs={**kwargs, "score": enc.score_ord, "config": config},
        label=f"GLMM/{label_prefix}",
        warmup=warmup,
        samples=samples,
        chains=chains,
        target_accept=target_accept,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
    )
    t_glmm = time.time() - t_glmm0

    var_lmm = gtheory.variance_table(samples_lmm, "LMM", residual_var=None)
    var_glmm = gtheory.variance_table(
        samples_glmm, "GLMM", residual_var=gtheory.LOGIT_RESIDUAL_VAR
    )
    var_df = pd.concat([var_lmm, var_glmm], ignore_index=True)
    var_df.to_csv(out_dir / "variance_components.csv", index=False)

    ds_lmm = gtheory.latent_dstudy(
        samples_lmm, residual_var=None, ni_grid=ni_grid, nj_grid=nj_grid
    )
    ds_lmm.insert(0, "model", "LMM")
    ds_glmm = gtheory.latent_dstudy(
        samples_glmm,
        residual_var=gtheory.LOGIT_RESIDUAL_VAR,
        ni_grid=ni_grid,
        nj_grid=nj_grid,
    )
    ds_glmm.insert(0, "model", "GLMM")
    ds_latent = pd.concat([ds_lmm, ds_glmm], ignore_index=True)
    ds_latent.to_csv(out_dir / "dstudy_latent.csv", index=False)

    ds_prob = pd.DataFrame()
    if not skip_prob_scale:
        cuts = np.asarray(samples_glmm["cutpoints"])
        ds_prob_lmm = gtheory.probability_dstudy(
            samples_lmm,
            k_sim_l,
            label="LMM",
            cutpoints_samples=None,
            ni_grid=ni_grid,
            nj_grid=nj_grid,
        )
        ds_prob_lmm.insert(0, "model", "LMM")
        ds_prob_glmm = gtheory.probability_dstudy(
            samples_glmm,
            k_sim_g,
            label="GLMM",
            cutpoints_samples=cuts,
            ni_grid=ni_grid,
            nj_grid=nj_grid,
        )
        ds_prob_glmm.insert(0, "model", "GLMM")
        ds_prob = pd.concat([ds_prob_lmm, ds_prob_glmm], ignore_index=True)
        ds_prob.to_csv(out_dir / "dstudy_probability.csv", index=False)

    gtheory.save_dstudy_samples(samples_lmm, out_dir / "samples_lmm.npz")
    gtheory.save_dstudy_samples(samples_glmm, out_dir / "samples_glmm.npz")

    return dict(
        var_df=var_df,
        ds_latent=ds_latent,
        ds_prob=ds_prob,
        meta=dict(
            n_models=enc.n_models,
            n_items=enc.n_items,
            n_judges=enc.n_judges,
            n_obs=len(df),
            t_lmm=t_lmm,
            t_glmm=t_glmm,
            div_lmm=div_lmm,
            div_glmm=div_glmm,
        ),
    )


def fit_het(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    config: gtheory.ModelConfig,
    ni_grid: Sequence[int],
    nj_grid: Sequence[int],
    label_prefix: str,
    seed_label: str,
    warmup: int,
    samples: int,
    chains: int,
    target_accept: float,
    dense_mass: bool,
    max_tree_depth: int,
    var_filename: str = "variance_components_het.csv",
) -> dict:
    enc = gtheory.encode(df)
    print(f"  obs={len(df)}  M={enc.n_models} I={enc.n_items} J={enc.n_judges}")

    kwargs = dict(
        model_idx=enc.model_idx,
        item_idx=enc.item_idx,
        judge_idx=enc.judge_idx,
        n_models=enc.n_models,
        n_items=enc.n_items,
        n_judges=enc.n_judges,
    )
    (k_het,) = _seed_key(seed_label, n=1)

    t0 = time.time()
    samples_het, _, div_het, _ = gtheory.run_nuts(
        gtheory.glmm_ordinal_het,
        k_het,
        model_kwargs={**kwargs, "score": enc.score_ord, "config": config},
        label=f"HET/{label_prefix}",
        warmup=warmup,
        samples=samples,
        chains=chains,
        target_accept=target_accept,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
    )
    t_het = time.time() - t0

    het = compute_het_artifacts(
        samples_het, enc.judges, ni_grid, nj_grid, out_dir=out_dir
    )
    avg_eff_res = het["avg_eff_res"]
    s_j_summary = het["s_j_summary"]
    ds_het = het["ds_het"]

    var_het = gtheory.variance_table(samples_het, "GLMM-het", residual_var=avg_eff_res)
    var_het.to_csv(out_dir / var_filename, index=False)
    s_j_summary.to_csv(out_dir / "s_j_summary.csv", index=False)
    ds_het.to_csv(out_dir / "dstudy_latent_het.csv", index=False)

    return dict(
        var_df=var_het,
        ds_het=ds_het,
        s_j_summary=s_j_summary,
        meta=dict(
            n_models=enc.n_models,
            n_items=enc.n_items,
            n_judges=enc.n_judges,
            n_obs=len(df),
            judges=enc.judges,
            t_het=t_het,
            div_het=div_het,
            s_j_median=s_j_summary["s_j_median"].tolist(),
            s_j_low=s_j_summary["s_j_low"].tolist(),
            s_j_high=s_j_summary["s_j_high"].tolist(),
            avg_eff_residual=avg_eff_res,
        ),
    )


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_max_tree_depth: int = 10,
) -> None:
    """--only/--limit are NOT added here because their types differ per script
    (WildBench int list vs HELM scenario-name filter)."""
    parser.add_argument("--warmup", type=int, default=gtheory.NUTS_WARMUP)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=gtheory.TARGET_ACCEPT)
    parser.add_argument(
        "--dense-mass", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-tree-depth", type=int, default=default_max_tree_depth)


def _insert_keys(df: pd.DataFrame, key: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for i, (k, v) in enumerate(key.items()):
        out.insert(i, k, v)
    return out


def _print_banner(spec: DatasetSpec, args, out_root: Path) -> None:
    print(f"[{spec.name}] {len(spec.cells)} cells; out_dir={out_root.resolve()}")
    print(
        f"[{spec.name}] NUTS: warmup={args.warmup} samples={args.samples} "
        f"chains={args.chains} target_accept={args.target_accept}"
    )


def _fit_kwargs(spec: DatasetSpec, args, cell_dir: Path) -> dict:
    return dict(
        out_dir=cell_dir,
        config=spec.config,
        warmup=args.warmup,
        samples=args.samples,
        chains=args.chains,
        target_accept=args.target_accept,
        dense_mass=args.dense_mass,
        max_tree_depth=args.max_tree_depth,
        ni_grid=spec.ni_grid,
        nj_grid=spec.nj_grid,
    )


def _run_sweep_inner(
    spec: DatasetSpec,
    args,
    *,
    is_het: bool,
    with_recommendations: bool = False,
) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    per_dir = out_root / spec.cell_subdir
    per_dir.mkdir(parents=True, exist_ok=True)

    numpyro.set_host_device_count(args.chains)
    np.random.seed(gtheory.SEED)

    _print_banner(spec, args, out_root)

    ds_rows: list[pd.DataFrame] = []
    var_rows: list[pd.DataFrame] = []
    sj_rows: list[pd.DataFrame] = []
    rec_rows: list[pd.DataFrame] = []
    log: list[dict] = []

    suffix = "_het" if is_het else ""
    t_overall = time.time()
    for i, cell in enumerate(spec.cells, 1):
        cell_dir = per_dir / cell.dirname
        cell_dir.mkdir(parents=True, exist_ok=True)
        label = " / ".join(str(v) for v in cell.key.values())
        print(f"\n[{i}/{len(spec.cells)}] {label}")
        t0 = time.time()
        try:
            res = spec.fit_cell(cell, **_fit_kwargs(spec, args, cell_dir))
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            log.append(
                {
                    **cell.key,
                    "status": "failed",
                    "error": str(e),
                    "seconds": time.time() - t0,
                }
            )
            (out_root / "sweep_log.txt").write_text(json.dumps(log, indent=2))
            continue

        dt = time.time() - t0
        m = res["meta"]
        log_row = {
            **cell.key,
            "status": "ok",
            "n_obs": m["n_obs"],
            "n_models": m["n_models"],
            "n_items": m["n_items"],
            "n_judges": m["n_judges"],
        }
        if is_het:
            log_row.update(
                {
                    "judges": m["judges"],
                    **cell.payload,
                    "s_j_median": m["s_j_median"],
                    "s_j_low": m["s_j_low"],
                    "s_j_high": m["s_j_high"],
                    "avg_eff_residual": m["avg_eff_residual"],
                    "t_het": m["t_het"],
                    "div_het": m["div_het"],
                    "seconds": dt,
                }
            )
            print(
                f"  OK in {dt:.1f}s  (NUTS HET={m['t_het']:.0f}s; div={m['div_het']}; "
                f"s_j={[round(x, 3) for x in m['s_j_median']]})"
            )
            ds_rows.append(_insert_keys(res["ds_het"], cell.key))
            var_rows.append(_insert_keys(res["var_df"], cell.key))
            sj_rows.append(_insert_keys(res["s_j_summary"], cell.key))
        else:
            log_row.update(
                {
                    **cell.payload,
                    "t_lmm": m["t_lmm"],
                    "t_glmm": m["t_glmm"],
                    "div_lmm": m["div_lmm"],
                    "div_glmm": m["div_glmm"],
                    "seconds": dt,
                }
            )
            print(
                f"  OK in {dt:.1f}s  (NUTS LMM={m['t_lmm']:.0f}s GLMM={m['t_glmm']:.0f}s; "
                f"div LMM={m['div_lmm']} GLMM={m['div_glmm']})"
            )
            for src_df in (res.get("ds_latent"), res.get("ds_prob")):
                if src_df is None or src_df.empty:
                    continue
                ds_rows.append(_insert_keys(src_df, cell.key))
            var_rows.append(_insert_keys(res["var_df"], cell.key))
            if with_recommendations and "rec_df" in res and not res["rec_df"].empty:
                rec_rows.append(_insert_keys(res["rec_df"], cell.key))
        log.append(log_row)

        if ds_rows:
            pd.concat(ds_rows, ignore_index=True).to_csv(
                out_root / f"aggregate_dstudy{suffix}.csv", index=False
            )
        if var_rows:
            pd.concat(var_rows, ignore_index=True).to_csv(
                out_root / f"aggregate_variance{suffix}.csv", index=False
            )
        if sj_rows:
            pd.concat(sj_rows, ignore_index=True).to_csv(
                out_root / "aggregate_s_j.csv", index=False
            )
        if rec_rows:
            pd.concat(rec_rows, ignore_index=True).to_csv(
                out_root / "aggregate_recommendations.csv", index=False
            )
        (out_root / "sweep_log.txt").write_text(json.dumps(log, indent=2))

    print(f"\n[done] total wall {time.time() - t_overall:.0f}s")

    if not is_het and ds_rows:
        ds_long = pd.concat(ds_rows, ignore_index=True)
        group_cols = list(spec.cells[0].key.keys())
        h = headline_max(ds_long, group_cols)
        h.to_csv(out_root / "headline_max_design.csv", index=False)
        gap = lmm_vs_glmm_gap(h, group_cols)
        gap.to_csv(out_root / "headline_lmm_vs_glmm_gap.csv", index=False)
        print(f"\n=== {spec.name} HEADLINE: LMM vs GLMM at max design ===")
        print(gap.to_string(index=False))


def run_vanilla_sweep(
    spec: DatasetSpec, args, *, with_recommendations: bool = False
) -> None:
    _run_sweep_inner(
        spec, args, is_het=False, with_recommendations=with_recommendations
    )


def run_het_sweep(spec: DatasetSpec, args) -> None:
    _run_sweep_inner(spec, args, is_het=True)
