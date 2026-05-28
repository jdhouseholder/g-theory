from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd


def cached_url_get(local_path: Path | str, url: str, *, timeout: int = 60) -> Path:
    local_path = Path(local_path)
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    local_path.write_bytes(data)
    return local_path


def print_long_form_summary(df: pd.DataFrame) -> None:
    print(f"Long-form rows: {len(df)}")
    print(f"  models: {df.model.nunique()}  {sorted(df.model.unique())}")
    print(f"  items: {df.item.nunique()}")
    print(f"  judges: {df.judge.nunique()}  {sorted(df.judge.unique())}")
    print(f"  score distribution: {dict(sorted(df.score_ord.value_counts().items()))}")
    n_M = df.model.nunique()
    n_I = df.item.nunique()
    n_J = df.judge.nunique()
    print(
        f"  filled cells: {len(df)} / max {n_M * n_I * n_J} "
        f"= {100 * len(df) / (n_M * n_I * n_J):.1f}% crossed"
    )
