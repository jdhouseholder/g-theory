from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def min_ni(df: pd.DataFrame, target: float, n_J: int, phi_col: str) -> int | None:
    s = df[df.n_J == n_J].sort_values("n_I")
    hit = s[s[phi_col] >= target]
    return int(hit.n_I.iloc[0]) if not hit.empty else None


def collect_one(
    name: str, df: pd.DataFrame, group_col: str | None, phi_col: str
) -> pd.DataFrame:
    targets = [0.70, 0.80, 0.90]
    n_js = [1, 2, 3, 5, 8]
    df = df[df.scale == "latent"].copy()
    if "n_J" not in df.columns:
        df = df.assign(n_J=1)
    groups = sorted(df[group_col].unique()) if group_col else [None]
    rows = []
    for g in groups:
        sub = df if g is None else df[df[group_col] == g]
        for m in sorted(sub.model.unique()):
            sm = sub[sub.model == m]
            for n_J in n_js:
                fit_here = n_J in sm.n_J.values
                for t in targets:
                    if fit_here:
                        mn = min_ni(sm, t, n_J, phi_col)
                        status = "feasible" if mn is not None else "infeasible"
                    else:
                        mn = None
                        status = "not_fit"
                    rows.append(
                        dict(
                            dataset=name,
                            group=str(g) if g is not None else "-",
                            model=m,
                            n_J=n_J,
                            target=t,
                            min_n_I=mn,
                            fit_status=status,
                        )
                    )
    return pd.DataFrame(rows)


def _build_long(phi_col: str) -> pd.DataFrame:
    sources = [
        ("HELM", "state/sweep_helm/aggregate_dstudy.csv", "criterion"),
        ("HELM", "state/sweep_helm_het/aggregate_dstudy_het.csv", "criterion"),
        ("BiGGen", "state/sweep_biggen/aggregate_dstudy.csv", "capability"),
        ("BiGGen", "state/sweep_biggen_het/aggregate_dstudy_het.csv", "capability"),
        ("WildBench", "state/sweep_wildbench/aggregate_dstudy.csv", "cap"),
        ("WildBench", "state/sweep_wildbench_het/aggregate_dstudy_het.csv", "cap"),
    ]
    pieces = []
    for name, path, gcol in sources:
        if not Path(path).exists():
            print(f"  missing: {path}")
            continue
        pieces.append(collect_one(name, pd.read_csv(path), gcol, phi_col))
    return pd.concat(pieces, ignore_index=True)


def _pivot_and_report(long: pd.DataFrame, label: str, out_root: Path) -> pd.DataFrame:
    long.to_csv(out_root / f"dstudy_recommendations_long_{label}.csv", index=False)
    keys = ["dataset", "group", "n_J", "target"]
    status_idx = long.pivot_table(
        index=keys, columns="model", values="fit_status", aggfunc="first"
    )
    wide_idx = long.pivot_table(
        index=keys, columns="model", values="min_n_I", aggfunc="first"
    ).reindex(status_idx.index)
    model_cols = [c for c in ["LMM", "GLMM", "GLMM-het"] if c in wide_idx.columns]

    def _spread(row, srow):
        vals = []
        any_infeasible = False
        for c in model_cols:
            s = srow[c]
            if s == "feasible":
                vals.append(row[c])
            elif s == "infeasible":
                any_infeasible = True
        if len(vals) < 1:
            return np.nan
        if any_infeasible:
            return np.inf
        if len(vals) < 2:
            return np.nan
        return max(vals) / min(vals)

    spread_vals = [
        _spread(wide_idx.loc[idx], status_idx.loc[idx]) for idx in wide_idx.index
    ]
    wide_idx = wide_idx.copy()
    wide_idx["spread"] = spread_vals
    for c in model_cols:
        wide_idx[f"{c}_status"] = status_idx[c]
    wide = wide_idx.reset_index()
    wide.to_csv(out_root / f"dstudy_recommendations_wide_{label}.csv", index=False)

    print(f"\n=== [{label}] Top disagreements: target=0.80, n_J in {{2,5,8}} ===")
    sub = wide[(wide.target == 0.80) & wide.n_J.isin([2, 5, 8])].sort_values(
        "spread", ascending=False, na_position="last"
    )
    print(
        sub[["dataset", "group", "n_J"] + model_cols + ["spread"]]
        .head(25)
        .to_string(index=False)
    )

    print(
        f"\n=== [{label}] Infeasibility at target=0.80, n_J=8 "
        "(model fit but never reaches target - excludes 'not_fit') ==="
    )
    n_J8 = wide[(wide.target == 0.80) & (wide.n_J == 8)]
    for m in model_cols:
        col = n_J8[f"{m}_status"]
        n_fit = (col != "not_fit").sum()
        n_inf = (col == "infeasible").sum()
        n_not_fit = (col == "not_fit").sum()
        print(
            f"  {m:<10} {n_inf:>3}/{n_fit} fit-and-infeasible "
            f"({n_not_fit} not fit on this slice)"
        )
    return wide


def main():
    out_root = Path("state")
    out_root.mkdir(exist_ok=True)

    long_median = _build_long("Phi_median")
    _pivot_and_report(long_median, "median", out_root)

    long_cilow = _build_long("Phi_low")
    _pivot_and_report(long_cilow, "cilow", out_root)

    # Back-compat: historical filenames (median criterion, no status columns).
    long_median.to_csv(out_root / "dstudy_recommendations_long.csv", index=False)
    wide_compat = long_median.pivot_table(
        index=["dataset", "group", "n_J", "target"],
        columns="model",
        values="min_n_I",
        aggfunc="first",
    ).reset_index()
    wide_compat.to_csv(out_root / "dstudy_recommendations_wide.csv", index=False)


if __name__ == "__main__":
    main()
