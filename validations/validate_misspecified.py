"""
validate_misspecified.py - factorial synthetic at n_M=4 to isolate the
mechanism behind the HELM/BiGGen LMM-vs-GLMM disagreement.

Tests four regimes crossing {neutral, ceiling-effect cutpoints} x
{homoscedastic, heteroscedastic judges} to determine which pathology
drives the LMM-GLMM gap.
"""

from __future__ import annotations
import argparse
import json
import time
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtheory


N_M = 4  # matches HELM/BiGGen
N_I = 100
N_J = 6

D_NI = 200
D_NJ = 8


# Matches a HELM-like profile (moderate sigma_M, sigma_MI; small sigma_I, sigma_J; tiny sigma_MJ, sigma_IJ).
BASE_SIGMAS = dict(
    M=1.0,
    I=0.5,
    J=0.3,
    MI=1.0,
    MJ=0.2,
    IJ=0.3,
)

CUT_NEUTRAL = np.array([-2.0, -0.5, 0.5, 2.0])
CUT_CEILING = np.array([-3.5, -2.0, -1.0, 0.0])

HOM_JUDGE = np.array([1.0] * N_J)
HET_JUDGE = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])

REGIMES = {
    "R_clean": dict(
        cutpoints=CUT_NEUTRAL,
        judge_noise=HOM_JUDGE,
        note="clean ordinal, homosc - sanity check",
    ),
    "R_ceiling": dict(
        cutpoints=CUT_CEILING,
        judge_noise=HOM_JUDGE,
        note="skewed cutpoints (ceiling effect)",
    ),
    "R_hetero": dict(
        cutpoints=CUT_NEUTRAL,
        judge_noise=HET_JUDGE,
        note="heteroscedastic judges (2 of 6 are 2x noisier)",
    ),
    "R_combo": dict(
        cutpoints=CUT_CEILING,
        judge_noise=HET_JUDGE,
        note="ceiling + heteroscedasticity (HELM Concise/Understand profile)",
    ),
}


def _generate(
    rng_key, regime: dict, n_M: int, n_I: int, n_J: int
) -> tuple[pd.DataFrame, dict]:
    s = BASE_SIGMAS
    cutpoints = regime["cutpoints"]
    judge_mult = regime["judge_noise"]
    keys = jax.random.split(rng_key, 8)
    alpha = jax.random.normal(keys[0], (n_M,)) * np.sqrt(s["M"])
    beta = jax.random.normal(keys[1], (n_I,)) * np.sqrt(s["I"])
    gamma = jax.random.normal(keys[2], (n_J,)) * np.sqrt(s["J"])
    ab = jax.random.normal(keys[3], (n_M, n_I)) * np.sqrt(s["MI"])
    ag = jax.random.normal(keys[4], (n_M, n_J)) * np.sqrt(s["MJ"])
    bg = jax.random.normal(keys[5], (n_I, n_J)) * np.sqrt(s["IJ"])
    eta = (
        alpha[:, None, None]
        + beta[None, :, None]
        + gamma[None, None, :]
        + ab[:, :, None]
        + ag[:, None, :]
        + bg[None, :, :]
    )
    base_noise = jax.random.logistic(keys[6], eta.shape)
    mult = jnp.asarray(judge_mult, dtype=base_noise.dtype)
    latent = eta + base_noise * mult[None, None, :]
    cuts = jnp.asarray(cutpoints)
    y_ord = jnp.sum(latent[..., None] > cuts, axis=-1)
    y_flat = (np.asarray(y_ord) + 1).reshape(-1)  # score 1..K
    m_idx, i_idx, j_idx = np.meshgrid(
        np.arange(n_M), np.arange(n_I), np.arange(n_J), indexing="ij"
    )
    df = pd.DataFrame(
        {
            "model": m_idx.reshape(-1).astype(str),
            "item": i_idx.reshape(-1).astype(str),
            "judge": j_idx.reshape(-1).astype(str),
            "score_ord": y_flat.astype(np.int64),
            "score_raw": y_flat.astype(np.float64),
        }
    )
    return df, dict(eta_var=float(eta.var()), latent_var=float(latent.var()))


def generate_one(rng_key, regime: dict) -> tuple[pd.DataFrame, dict]:
    return _generate(rng_key, regime, N_M, N_I, N_J)


def truth_phi_latent(regime: dict, n_I=D_NI, n_J=D_NJ) -> float:
    """Population-level latent Phi treating judge noise as having an averaged variance."""
    s = BASE_SIGMAS
    avg_noise_var = np.mean(np.asarray(regime["judge_noise"]) ** 2) * (np.pi**2 / 3.0)
    sigma_delta = s["MI"] / n_I + s["MJ"] / n_J + avg_noise_var / (n_I * n_J)
    sigma_Delta = sigma_delta + s["I"] / n_I + s["J"] / n_J + s["IJ"] / (n_I * n_J)
    return s["M"] / (s["M"] + sigma_Delta)


def fit_models_phi(
    df: pd.DataFrame,
    *,
    model_set,
    rep_seed: int,
    warmup: int,
    samples: int,
    chains: int,
    target_accept: float,
    alpha_centered: float,
    n_I: int = D_NI,
    n_J: int = D_NJ,
) -> dict:
    """Fit any subset of {"LMM","GLMM","HET"} on df, return Phi at (n_I, n_J)."""
    config = gtheory.default_config().with_overrides(
        reparam={"alpha": float(alpha_centered)}
    )
    enc = gtheory.encode(df)
    kw = dict(
        model_idx=enc.model_idx,
        item_idx=enc.item_idx,
        judge_idx=enc.judge_idx,
        n_models=enc.n_models,
        n_items=enc.n_items,
        n_judges=enc.n_judges,
    )
    model_set = list(model_set)
    keys = jax.random.split(jax.random.PRNGKey(rep_seed), len(model_set))
    out: dict = {}
    for key, name in zip(keys, model_set):
        t0 = time.time()
        if name == "LMM":
            samp, _, div, _ = gtheory.run_nuts(
                gtheory.lmm_model,
                key,
                model_kwargs={**kw, "score": enc.score_raw, "config": config},
                label=f"LMM/rep{rep_seed}",
                warmup=warmup,
                samples=samples,
                chains=chains,
                target_accept=target_accept,
            )
            ds = gtheory.latent_dstudy(
                samp, residual_var=None, ni_grid=[n_I], nj_grid=[n_J]
            )
        elif name == "GLMM":
            samp, _, div, _ = gtheory.run_nuts(
                gtheory.glmm_ordinal,
                key,
                model_kwargs={**kw, "score": enc.score_ord, "config": config},
                label=f"GLMM/rep{rep_seed}",
                warmup=warmup,
                samples=samples,
                chains=chains,
                target_accept=target_accept,
            )
            ds = gtheory.latent_dstudy(
                samp,
                residual_var=gtheory.LOGIT_RESIDUAL_VAR,
                ni_grid=[n_I],
                nj_grid=[n_J],
            )
        elif name == "HET":
            samp, _, div, _ = gtheory.run_nuts(
                gtheory.glmm_ordinal_het,
                key,
                model_kwargs={**kw, "score": enc.score_ord, "config": config},
                label=f"HET/rep{rep_seed}",
                warmup=warmup,
                samples=samples,
                chains=chains,
                target_accept=target_accept,
            )
            s_j = np.asarray(samp["s_j"])
            eff_res = gtheory.LOGIT_RESIDUAL_VAR * (s_j**2).mean(axis=-1)
            ds = gtheory.latent_dstudy(
                samp, residual_var=eff_res, ni_grid=[n_I], nj_grid=[n_J]
            )
            out["s_j_median"] = np.median(s_j, axis=0).tolist()
        else:
            raise ValueError(f"Unknown model name {name!r}")
        prefix = name.lower()
        out[f"{prefix}_phi"] = float(ds["Phi_median"].iloc[0])
        out[f"{prefix}_phi_lo"] = float(ds["Phi_low"].iloc[0])
        out[f"{prefix}_phi_hi"] = float(ds["Phi_high"].iloc[0])
        out[f"div_{prefix}"] = div
        out[f"t_{prefix}"] = time.time() - t0
    return out


def fit_one(
    df: pd.DataFrame,
    *,
    rep_seed: int,
    warmup: int,
    samples: int,
    chains: int,
    target_accept: float,
    alpha_centered: float,
) -> dict:
    return fit_models_phi(
        df,
        model_set=("LMM", "GLMM"),
        rep_seed=rep_seed,
        warmup=warmup,
        samples=samples,
        chains=chains,
        target_accept=target_accept,
        alpha_centered=alpha_centered,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=None,
        help="Subset of regimes to run (default: all 4).",
    )
    parser.add_argument("--warmup", type=int, default=gtheory.NUTS_WARMUP)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=gtheory.TARGET_ACCEPT)
    parser.add_argument(
        "--alpha-centered",
        type=float,
        default=1.0,
        help="config.reparam['alpha']: 1.0=centered (best for big sigma_M).",
    )
    parser.add_argument("--out-dir", default="./state/validate_misspec_out")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numpyro.set_host_device_count(args.chains)
    np.random.seed(gtheory.SEED)

    regimes = REGIMES if not args.regimes else {k: REGIMES[k] for k in args.regimes}
    print(f"[misspec] design = {N_M}M x {N_I}I x {N_J}J  D-study at ({D_NI},{D_NJ})")
    print(f"[misspec] reps={args.reps}  regimes={list(regimes.keys())}")
    print(f"[misspec] alpha_centered={args.alpha_centered}\n")

    all_rows = []
    summary = {}
    for reg_name, reg in regimes.items():
        truth = truth_phi_latent(reg)
        reg_dir = out / reg_name
        reg_dir.mkdir(exist_ok=True)
        print(f"=== {reg_name} : {reg['note']} ===")
        print(f"    cutpoints={reg['cutpoints']}")
        print(f"    judge_noise={reg['judge_noise']}")
        print(f"    truth Phi(latent, {D_NI},{D_NJ}) = {truth:.4f}")
        reg_rows = []
        for rep in range(args.reps):
            seed = gtheory.SEED + zlib.crc32(reg_name.encode()) % 100000 + rep
            print(f"  --- rep {rep + 1}/{args.reps} (seed={seed}) ---")
            k = jax.random.PRNGKey(seed)
            df, gen_meta = generate_one(k, reg)
            score_dist = dict(sorted(df.score_ord.value_counts().items()))
            print(f"    score dist: {score_dist}")
            r = fit_one(
                df,
                rep_seed=seed,
                warmup=args.warmup,
                samples=args.samples,
                chains=args.chains,
                target_accept=args.target_accept,
                alpha_centered=args.alpha_centered,
            )
            r.update(dict(regime=reg_name, rep=rep, seed=seed, truth=truth))
            r["lmm_bias"] = r["lmm_phi"] - truth
            r["glmm_bias"] = r["glmm_phi"] - truth
            r["lmm_minus_glmm"] = r["lmm_phi"] - r["glmm_phi"]
            print(
                f"    LMM Phi  = {r['lmm_phi']:.3f}  bias={r['lmm_bias']:+.3f}  div={r['div_lmm']}"
            )
            print(
                f"    GLMM Phi = {r['glmm_phi']:.3f}  bias={r['glmm_bias']:+.3f}  div={r['div_glmm']}"
            )
            print(f"    LMM-GLMM = {r['lmm_minus_glmm']:+.3f}")
            all_rows.append(r)
            reg_rows.append(r)
        rdf = pd.DataFrame(reg_rows)
        # clean_mean excludes reps with pathological divergence counts where
        # the GLMM posterior likely failed to identify sigma_M.
        DIV_THRESH = 5
        clean = rdf[rdf.div_glmm <= DIV_THRESH]
        summary[reg_name] = dict(
            truth=truth,
            lmm_mean_phi=float(rdf.lmm_phi.mean()),
            lmm_mean_bias=float(rdf.lmm_bias.mean()),
            glmm_mean_phi=float(rdf.glmm_phi.mean()),
            glmm_mean_bias=float(rdf.glmm_bias.mean()),
            glmm_median_bias=float(rdf.glmm_bias.median()),
            glmm_clean_mean_bias=(
                float(clean.glmm_bias.mean()) if len(clean) else None
            ),
            lmm_minus_glmm_mean=float(rdf.lmm_minus_glmm.mean()),
            lmm_minus_glmm_median=float(rdf.lmm_minus_glmm.median()),
            frac_lmm_gt_glmm=float((rdf.lmm_minus_glmm > 0).mean()),
            n_reps=len(rdf),
            n_reps_clean=int(len(clean)),
            div_glmm_max=int(rdf.div_glmm.max()),
        )
        rdf.to_csv(reg_dir / "results.csv", index=False)
        print(
            f"  --> regime mean LMM-GLMM = {summary[reg_name]['lmm_minus_glmm_mean']:+.3f}"
        )
        print(
            f"      ({summary[reg_name]['frac_lmm_gt_glmm'] * 100:.0f}% LMM > GLMM)\n"
        )
        pd.DataFrame(all_rows).to_csv(out / "results.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== FINAL SUMMARY ===")
    for reg_name, s in summary.items():
        print(
            f"{reg_name:14s}  truth={s['truth']:.3f}  "
            f"LMM-GLMM={s['lmm_minus_glmm_mean']:+.4f}  "
            f"({s['frac_lmm_gt_glmm'] * 100:.0f}% pos)"
        )
    print(f"\nOutputs -> {out.resolve()}")


if __name__ == "__main__":
    main()
