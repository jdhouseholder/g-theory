"""figure_heteroscedasticity.py - Per-judge s_j heatmap on BiGGen-Bench.

Writes to ./state/figures/figure_heteroscedasticity.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from _style import COLORS, load_headline, save_figure
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


OUT = Path("./state/figures/figure_heteroscedasticity.png")
PER_CAP = Path("./state/sweep_biggen_het/per_capability")
BIGGEN = Path("./state/sweep_biggen")

JUDGES = [
    "human", "claude", "gpt4", "gpt4_turbo",
    "prometheus_8x7b", "prometheus_8x7b_bgb",
]
JUDGE_LABELS = [
    "human", "claude", "gpt-4", "gpt-4-turbo",
    "prometheus-8x7b", "prometheus-8x7b-bgb",
]


def main():
    # Order capabilities by LMM-GLMM Phi gap so the largest-gap cell is leftmost.
    gap = load_headline(BIGGEN).sort_values("Phi_gap_median", ascending=True)
    caps = gap["capability"].tolist()

    M = np.zeros((len(JUDGES), len(caps)))
    for ci, cap in enumerate(caps):
        sj = pd.read_csv(PER_CAP / cap / "s_j_summary.csv")
        for ji, j in enumerate(JUDGES):
            M[ji, ci] = sj.loc[sj.judge == j, "s_j_median"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    # Diverging norm centered at 1 (the homoscedastic baseline s_j = 1).
    norm = TwoSlopeNorm(vmin=0.3, vcenter=1.0, vmax=2.7)
    im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")

    for ji in range(len(JUDGES)):
        for ci in range(len(caps)):
            v = M[ji, ci]
            color = "white" if (v > 1.6 or v < 0.55) else "black"
            ax.text(ci, ji, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8.5)

    ax.set_yticks(range(len(JUDGES)))
    ax.set_yticklabels(JUDGE_LABELS, fontsize=9)
    # Highlight the human row - it's the canonical "noisy" reference.
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color(COLORS["human"])
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels([c.replace("_", " ") for c in caps],
                       fontsize=8, rotation=30, ha="right")
    ax.set_title(
        "Per-judge logistic scale $s_j$ on BiGGen-Bench\n"
        "(red = noisier than the homoscedastic $s_j = 1$ assumption)",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("$s_j$", fontsize=9)
    cbar.ax.axhline(1.0, color="black", lw=1)

    fig.tight_layout()
    save_figure(fig, OUT)


if __name__ == "__main__":
    main()
