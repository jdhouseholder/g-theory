from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from loaders import print_long_form_summary

HF_REPO = "prometheus-eval/BiGGen-Bench-Results"
HF_FILE = "data/human_eval-00000-of-00001.parquet"

LOCAL_PATH = Path(__file__).parent.parent / "datasets" / "biggen_data" / "human_eval.parquet"

CAPABILITIES = [
    "grounding",
    "instruction_following",
    "reasoning",
    "theory_of_mind",
    "safety",
    "refinement",
    "planning",
    "tool_usage",
]

JUDGE_COLS = {
    "human": ("human_score", "scalar"),
    "prometheus_8x7b": ("prometheus_8x7b_score", "array"),
    "prometheus_8x7b_bgb": ("prometheus_8x7b_bgb_score", "array"),
    "gpt4": ("gpt4_score", "scalar"),
    "gpt4_turbo": ("gpt4_04_turbo_score", "scalar"),
    "claude": ("claude_score", "scalar"),
}


def fetch_all(verbose: bool = True) -> Path:
    if LOCAL_PATH.exists():
        if verbose:
            print(f"[biggen] cached at {LOCAL_PATH}")
        return LOCAL_PATH
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    LOCAL_PATH.write_bytes(Path(src).read_bytes())
    if verbose:
        print(f"[biggen] pulled {HF_FILE} -> {LOCAL_PATH}")
    return LOCAL_PATH


def load_biggen(capability: str | None = None) -> pd.DataFrame:
    if not LOCAL_PATH.exists():
        fetch_all(verbose=False)
    df_raw = pd.read_parquet(LOCAL_PATH)
    if capability is not None:
        df_raw = df_raw[df_raw["capability"] == capability]
        if df_raw.empty:
            raise ValueError(f"No rows for capability={capability!r}")

    rows = []
    for _, r in df_raw.iterrows():
        for judge_name, (col, kind) in JUDGE_COLS.items():
            raw = r[col]
            if kind == "array":
                if raw is None:
                    continue
                vals = np.asarray(raw, dtype=float)
                vals = vals[vals >= 1]  # -1 is BiGGen's parse-fail marker
                if len(vals) == 0:
                    continue
                score_raw = float(vals.mean())
            else:
                if raw is None or pd.isna(raw):
                    continue
                score_raw = float(raw)
            if score_raw < 1 or score_raw > 5:
                continue
            score_ord = int(round(score_raw))
            score_ord = max(1, min(5, score_ord))
            rows.append((r["model_name"], r["id"], judge_name, score_ord, score_raw))

    return pd.DataFrame(
        rows, columns=["model", "item", "judge", "score_ord", "score_raw"]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capability",
        default=None,
        help="Subset to one capability (e.g. 'reasoning'). "
        "Default: all 8 capabilities pooled.",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Download human_eval.parquet (idempotent).",
    )
    args = parser.parse_args()

    if args.prefetch:
        fetch_all()

    df = load_biggen(args.capability)
    print_long_form_summary(df)
