"""Shared helpers for figure scripts.

Importing this module sets the matplotlib Agg backend, so figure scripts can
do `from _style import ...` once and then `import matplotlib.pyplot as plt`
without their own `matplotlib.use("Agg")` calls.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless backend; must run before any pyplot import


COLORS: dict[str, str] = {
    "LMM": "#1f77b4",
    "GLMM": "#d62728",
    "GLMM_het": "#9467bd",
    "human": "#a00",
    "signal": "#2ca02c",
    "biggen": "#ff7f0e",
}


def save_figure(fig, path: Path | str, *, dpi: int = 140) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Figure -> {path}")
    return path


def load_headline(
    sweep_dir: Path | str, *, scale: str | None = "latent"
) -> pd.DataFrame:
    df = pd.read_csv(Path(sweep_dir) / "headline_lmm_vs_glmm_gap.csv")
    if scale is not None:
        df = df[df["scale"] == scale]
    return df
