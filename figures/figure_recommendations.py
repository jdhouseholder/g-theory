"""figure_recommendations.py - Headline D-study recommendation figure.

Writes to ./state/figures/figure_recommendations.png. Reads
state/dstudy_recommendations_wide.csv and per-sweep sweep_log.txt files.

Marker convention:
  Solid    converged fit (< DIV_THRESHOLD divergences) reaching target
  Hollow   divergent posterior at same position (recommendation not interpretable)
  Infeas   model never reaches target on the grid (right-edge column)
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _style import COLORS, save_figure
import matplotlib.pyplot as plt


OUT = Path("./state/figures/figure_recommendations.png")
TARGET = 0.9
INF_X = 8000  # x-position for the "infeasible" column
DIV_THRESHOLD = 500  # >= 500 divergences (of 4000 samples) flags the fit
MODELS = [
    ("LMM", COLORS["LMM"], "o"),
    ("GLMM", COLORS["GLMM"], "s"),
    ("GLMM-het", COLORS["GLMM_het"], "D"),
]

# WildBench vanilla GLMM is ~100% divergent at every n_M; hardcoded because
# the current sweep_log.txt is a per_cap_reagg stub without div counts.
WILDBENCH_GLMM_DIV = 4000


def _load_divs(path: Path, key_field) -> dict:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return {}
    out = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        div = r.get("div_glmm")
        if div is None:
            continue
        if isinstance(key_field, tuple):
            key = tuple(r.get(f) for f in key_field)
            if any(k is None for k in key):
                continue
        else:
            key = r.get(key_field)
            if key is None:
                continue
        out[str(key)] = int(div)
    return out


def divergent_glmm(dataset: str, group: str) -> bool:
    if dataset == "WildBench":
        return WILDBENCH_GLMM_DIV >= DIV_THRESHOLD
    if dataset == "BiGGen":
        divs = _load_divs(Path("state/sweep_biggen/sweep_log.txt"), "capability")
        return divs.get(group, 0) >= DIV_THRESHOLD
    if dataset == "HELM":
        # HELM groups in the recommendations CSV are per-criterion (aggregated
        # across scenarios). Mark divergent if any scenario for the criterion was.
        divs = _load_divs(
            Path("state/sweep_helm/sweep_log.txt"), ("scenario", "criterion")
        )
        return any(
            v >= DIV_THRESHOLD for k, v in divs.items() if k[1] == group
        )
    return False


def cheapest_cost(wide: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, group), min over n_J of n_J * min_n_I at the target."""
    rows = []
    for (dataset, group), grp in wide.groupby(["dataset", "group"]):
        rec = {"dataset": dataset, "group": str(group)}
        for model, _, _ in MODELS:
            cands = [
                int(r.n_J) * int(r[model])
                for _, r in grp.iterrows()
                if pd.notna(r[model])
            ]
            rec[model] = min(cands) if cands else None
        rec["glmm_divergent"] = divergent_glmm(dataset, str(group))
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv("state/dstudy_recommendations_wide.csv")
    d = d[d.target == TARGET]
    cost = cheapest_cost(d)
    cost = cost.sort_values(["dataset", "group"]).reset_index(drop=True)

    def _label(row):
        if row["dataset"] == "WildBench":
            return f"WildBench: $n_M={row['group']}$"
        return f"{row['dataset']}: {row['group']}"

    cost["label"] = cost.apply(_label, axis=1)

    n_rows = len(cost)
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * n_rows + 1.2))

    y = np.arange(n_rows)

    # Connector lines between converged points (skip divergent vanilla GLMM).
    for i, row in cost.iterrows():
        xs = []
        for m, _, _ in MODELS:
            if pd.isna(row[m]):
                continue
            if m == "GLMM" and row["glmm_divergent"]:
                continue
            xs.append(row[m])
        if len(xs) >= 2:
            ax.plot(
                [min(xs), max(xs)], [i, i],
                color="grey", alpha=0.35, lw=1.0, zorder=1,
            )

    for model, color, marker in MODELS:
        feas_solid_x, feas_solid_y = [], []
        feas_hollow_x, feas_hollow_y = [], []
        inf_y = []
        for i, row in cost.iterrows():
            v = row[model]
            if pd.isna(v):
                inf_y.append(i)
                continue
            if model == "GLMM" and row["glmm_divergent"]:
                feas_hollow_x.append(v)
                feas_hollow_y.append(i)
            else:
                feas_solid_x.append(v)
                feas_solid_y.append(i)
        ax.scatter(
            feas_solid_x, feas_solid_y, color=color, marker=marker, s=55,
            zorder=3, label=model, edgecolor="white", linewidth=0.6,
        )
        # Hollow same-position marker for divergent fits.
        ax.scatter(
            feas_hollow_x, feas_hollow_y,
            facecolor="none", edgecolor=color, marker=marker, s=55,
            zorder=3, linewidth=1.4,
        )
        ax.scatter(
            [INF_X] * len(inf_y), inf_y,
            facecolor="none", edgecolor=color, marker=marker, s=55,
            zorder=3, linewidth=1.3, alpha=0.5,
        )

    ax.scatter(
        [], [], facecolor="none", edgecolor="black", marker="s", s=55,
        linewidth=1.4, label="divergent posterior",
    )

    ax.axvline(4000, color="grey", lw=0.6, ls=":", alpha=0.6)
    ax.text(
        INF_X, n_rows - 0.4, "infeasible",
        ha="center", va="bottom", fontsize=8, style="italic", color="grey",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(cost.label.tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, INF_X])
    ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "$\\infty$"])
    ax.set_xlim(40, 11000)
    ax.set_xlabel(
        f"Cheapest budget $n_I \\cdot n_J$ for $\\Phi \\geq {TARGET}$ (log scale)",
        fontsize=10,
    )
    ax.set_title(
        f"D-study recommendations disagree across likelihoods at $\\Phi \\geq {TARGET}$",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    save_figure(fig, OUT)


if __name__ == "__main__":
    main()
