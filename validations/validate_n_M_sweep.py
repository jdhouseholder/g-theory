"""
validate_n_M_sweep.py - does the LMM-GLMM gap predictably shrink with n_M?

Sweep n_M in {4, 6, 8, 10, 15, 20} on R_hetero synthetic with 3 reps each
and measure LMM-GLMM gap. A smooth dose-response supports the claim that
the gap is driven by poor identification of sigma_M at small n_M.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import jax
import numpyro
import pandas as pd

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtheory
from validations import validate_misspecified as vm


N_M_VALUES = [4, 6, 8, 10, 15, 20]
N_I = 60  # smaller items to keep each run fast at high n_M
N_J = 6


def generate_with_n_M(rng_key, regime, n_m):
    df, _ = vm._generate(rng_key, regime, n_m, N_I, N_J)
    return df


def fit_pair(df, *, rep_seed, warmup, samples, chains, target_accept, alpha_centered):
    return vm.fit_models_phi(
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
    parser.add_argument("--regime", default="R_hetero")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=gtheory.NUTS_WARMUP)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=gtheory.TARGET_ACCEPT)
    parser.add_argument("--alpha-centered", type=float, default=1.0)
    parser.add_argument("--out-dir", default="./state/validate_n_M_sweep_out")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numpyro.set_host_device_count(args.chains)

    reg = vm.REGIMES[args.regime]
    print(f"[n_M_sweep] regime={args.regime} reps={args.reps}")

    rows = []
    for n_m in N_M_VALUES:
        for rep in range(args.reps):
            seed = gtheory.SEED + n_m * 1000 + rep
            k = jax.random.PRNGKey(seed)
            df = generate_with_n_M(k, reg, n_m)
            t0 = time.time()
            r = fit_pair(
                df,
                rep_seed=seed,
                warmup=args.warmup,
                samples=args.samples,
                chains=args.chains,
                target_accept=args.target_accept,
                alpha_centered=args.alpha_centered,
            )
            dt = time.time() - t0
            gap = r["lmm_phi"] - r["glmm_phi"]
            r.update(
                dict(
                    regime=args.regime,
                    n_M=n_m,
                    rep=rep,
                    seed=seed,
                    lmm_minus_glmm=gap,
                    dt=dt,
                )
            )
            rows.append(r)
            print(
                f"  n_M={n_m} rep{rep + 1}: LMM={r['lmm_phi']:.3f}, "
                f"GLMM={r['glmm_phi']:.3f}, gap={gap:+.3f}  ({dt:.0f}s)"
            )
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== n_M MEANS ===")
    summ = df.groupby("n_M").agg(
        {
            "lmm_phi": ["mean", "std"],
            "glmm_phi": ["mean", "std"],
            "lmm_minus_glmm": ["mean", "std"],
        }
    )
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    print(summ.to_string(index=False))
    summ.to_csv(out / "summary.csv", index=False)
    print(f"\nOutputs -> {out.resolve()}")


if __name__ == "__main__":
    main()
