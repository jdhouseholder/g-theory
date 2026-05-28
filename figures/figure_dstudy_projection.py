"""figure_dstudy_projection.py - D-study Phi projection aggregated by dataset.

Writes to ./state/figures/figure_dstudy_projection.png.
"""

from pathlib import Path
import pandas as pd

from _style import COLORS, save_figure
import matplotlib.pyplot as plt


OUT = Path("./state/figures/figure_dstudy_projection.png")
TARGET = 0.9
N_J = 2

MODELS = [
    ("LMM",       COLORS["LMM"]),
    ("GLMM",      COLORS["GLMM"]),
    ("GLMM-het",  COLORS["GLMM_het"]),
]


def _concat_biggen():
    rows = []
    for sweep, suffix in [("sweep_biggen", "dstudy_latent.csv"),
                          ("sweep_biggen_het", "dstudy_latent_het.csv")]:
        base = Path(f"./state/{sweep}/per_capability")
        for cell in sorted(base.iterdir()):
            d = pd.read_csv(cell / suffix)
            d["cell"] = cell.name
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def _concat_helm():
    v = pd.read_csv("./state/sweep_helm/aggregate_dstudy.csv")
    v["cell"] = v["scenario"] + ":" + v["criterion"]
    h = pd.read_csv("./state/sweep_helm_het/aggregate_dstudy_het.csv")
    h["cell"] = h["scenario"] + ":" + h["criterion"]
    return pd.concat([v, h], ignore_index=True)


def _concat_wildbench():
    v = pd.read_csv("./state/sweep_wildbench/aggregate_dstudy.csv")
    v["cell"] = v["cap"].astype(str)
    h = pd.read_csv("./state/sweep_wildbench_het/aggregate_dstudy_het.csv")
    h["cell"] = h["cap"].astype(str)
    return pd.concat([v, h], ignore_index=True)


def _median_iqr(df, model):
    sub = df[(df.model == model) & (df.n_J == N_J)]
    if sub.empty:
        return None
    g = sub.groupby("n_I")["Phi_median"]
    return pd.DataFrame({
        "n_I": g.median().index.values,
        "median": g.median().values,
        "q25": g.quantile(0.25).values,
        "q75": g.quantile(0.75).values,
    }).sort_values("n_I")


def main():
    panels = [
        ("HELM Instruct (29 cells)", _concat_helm(),       False),
        ("BiGGen-Bench (8 caps)",    _concat_biggen(),     False),
        ("WildBench (4 caps)",       _concat_wildbench(),  True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)

    for ax, (title, df, dash_vanilla) in zip(axes, panels):
        per_model_max = []
        for model, color in MODELS:
            d = _median_iqr(df, model)
            if d is None or d.empty:
                continue
            per_model_max.append(d.n_I.max())
            # WildBench vanilla GLMM is divergent; dash + relabel so it's not read as a real recommendation.
            ls = "--" if (dash_vanilla and model == "GLMM") else "-"
            label = f"{model} (divergent)" if (dash_vanilla and model == "GLMM") else model
            ax.plot(d.n_I, d["median"], color=color, lw=2.0, ls=ls,
                    label=label, zorder=3)
            ax.fill_between(d.n_I, d.q25, d.q75, color=color, alpha=0.15, zorder=1)
        ax.axhline(TARGET, color="black", ls=":", lw=0.8, alpha=0.7)
        ax.set_xlabel(r"$n_I$ (items per judge)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        if per_model_max:
            ax.set_xlim(left=0, right=min(per_model_max))
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

    axes[0].set_ylabel(r"$\Phi$ (dependability)", fontsize=10)
    axes[0].text(
        axes[0].get_xlim()[0], TARGET + 0.015, fr"$\Phi={TARGET}$ target",
        fontsize=8, color="black", alpha=0.7,
    )

    fig.suptitle(
        fr"D-study projection at $n_J={N_J}$: median across cells, IQR shaded",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, OUT)


if __name__ == "__main__":
    main()
