"""
validate_het_on_misspec.py - does the het GLMM recover truth in R_hetero?

The vanilla GLMM shows consistent downward bias on small-n_M synthetic
data even when the generator is clean. This script tests whether the
heteroscedastic extension (per-judge logistic residual scale s_j) closes
the gap or inherits the same shrinkage.
"""

from __future__ import annotations
import argparse
import zlib
from pathlib import Path

import jax
import numpyro
import pandas as pd

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtheory
from validations import validate_misspecified as vm


def fit_three(df, *, rep_seed, warmup, samples, chains, target_accept, alpha_centered):
    return vm.fit_models_phi(
        df,
        model_set=("LMM", "GLMM", "HET"),
        rep_seed=rep_seed,
        warmup=warmup,
        samples=samples,
        chains=chains,
        target_accept=target_accept,
        alpha_centered=alpha_centered,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regimes", nargs="*", default=["R_clean", "R_hetero", "R_combo"]
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=gtheory.NUTS_WARMUP)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=gtheory.TARGET_ACCEPT)
    parser.add_argument("--alpha-centered", type=float, default=1.0)
    parser.add_argument("--out-dir", default="./state/validate_het_on_misspec_out")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numpyro.set_host_device_count(args.chains)

    rows = []
    for reg_name in args.regimes:
        reg = vm.REGIMES[reg_name]
        truth = vm.truth_phi_latent(reg)
        print(f"\n=== {reg_name} : truth Phi = {truth:.3f} ===")
        for rep in range(args.reps):
            seed = gtheory.SEED + zlib.crc32(reg_name.encode()) % 100000 + rep
            k = jax.random.PRNGKey(seed)
            df, _ = vm.generate_one(k, reg)
            r = fit_three(
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
            r["het_bias"] = r["het_phi"] - truth
            rows.append(r)
            print(f"  rep {rep + 1}/{args.reps}:")
            print(
                f"    LMM  Phi = {r['lmm_phi']:.3f}  bias={r['lmm_bias']:+.3f}  div={r['div_lmm']}"
            )
            print(
                f"    GLMM Phi = {r['glmm_phi']:.3f}  bias={r['glmm_bias']:+.3f}  div={r['div_glmm']}"
            )
            print(
                f"    HET  Phi = {r['het_phi']:.3f}  bias={r['het_bias']:+.3f}  div={r['div_het']}"
            )
            print(f"    s_j      = {[round(s, 2) for s in r['s_j_median']]}")
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== REGIME MEANS ===")
    print(
        df.groupby("regime")[["lmm_bias", "glmm_bias", "het_bias"]].mean().to_string()
    )
    summary = (
        df.groupby("regime")
        .agg(
            {
                "truth": "first",
                "lmm_phi": "mean",
                "lmm_bias": "mean",
                "glmm_phi": "mean",
                "glmm_bias": "mean",
                "het_phi": "mean",
                "het_bias": "mean",
            }
        )
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(summary.to_json(orient="records", indent=2))
    print(f"\nOutputs -> {out.resolve()}")


if __name__ == "__main__":
    main()
