from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from loaders import cached_url_get, print_long_form_summary

WILDBENCH_DIR = Path(__file__).parent.parent / "datasets" / "wildbench"
RELEASE = "v2.0522"
GH_BASE = (
    "https://raw.githubusercontent.com/allenai/WildBench/main/eval_results"
    f"/{RELEASE}/score.v2"
)

JUDGES = ["gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13"]

MODELS_INTERSECTION = [
    "Hermes-2-Theta-Llama-3-8B",
    "Llama-2-70b-chat-hf",
    "Llama-2-7b-chat-hf",
    "Llama-3-Instruct-8B-SimPO",
    "Llama-3-Instruct-8B-SimPO-ExPO",
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.2",
    "Mixtral-8x7B-Instruct-v0.1",
    "Nous-Hermes-2-Mixtral-8x7B-DPO",
    "Phi-3-medium-128k-instruct",
    "Phi-3-mini-128k-instruct",
    "Qwen1.5-72B-Chat",
    "Qwen1.5-72B-Chat-greedy",
    "Qwen1.5-7B-Chat@together",
    "Qwen2-72B-Instruct",
    "SELM-Zephyr-7B-iter-3",
    "Starling-LM-7B-beta",
    "Starling-LM-7B-beta-ExPO",
    "Yi-1.5-34B-Chat",
    "Yi-1.5-6B-Chat",
    "Yi-1.5-9B-Chat",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "command-r",
    "command-r-plus",
    "dbrx-instruct@together",
    "deepseekv2-chat",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemma-2b-it",
    "gemma-7b-it",
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "mistral-large-2402",
    "reka-flash-20240226",
    "tulu-2-dpo-70b",
    "yi-large",
]


def _local_path(judge: str, model: str) -> Path:
    return WILDBENCH_DIR / f"eval={judge}" / f"{model}.json"


def _download(judge: str, model: str) -> Path:
    return cached_url_get(
        _local_path(judge, model),
        f"{GH_BASE}/eval={judge}/{model}.json",
    )


def fetch_all(verbose: bool = True) -> None:
    for judge in JUDGES:
        for model in MODELS_INTERSECTION:
            p = _local_path(judge, model)
            existed = p.exists()
            _download(judge, model)
            if verbose and not existed:
                print(f"  pulled {judge}/{model}.json")
    if verbose:
        n = sum(
            1 for j in JUDGES for m in MODELS_INTERSECTION if _local_path(j, m).exists()
        )
        print(f"[wildbench_loader] {n} score files cached in {WILDBENCH_DIR}")


def load_wildbench(models: list[str] | None = None) -> pd.DataFrame:
    if models is None:
        models = MODELS_INTERSECTION
    rows = []
    for model in models:
        for judge in JUDGES:
            local = _download(judge, model)
            with open(local) as f:
                data = json.load(f)
            for rec in data:
                sid = rec["session_id"]
                raw = rec.get("score")
                if raw is None:
                    continue
                try:
                    s = int(str(raw).strip())
                except (TypeError, ValueError):
                    continue
                if s < 1 or s > 10:
                    continue
                rows.append((model, sid, judge, s, float(s)))
    return pd.DataFrame(
        rows, columns=["model", "item", "judge", "score_ord", "score_raw"]
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Download all 82 JSON files (idempotent).",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=None,
        help="Subsample to this many models (deterministic order).",
    )
    args = parser.parse_args()

    if args.prefetch:
        fetch_all()

    models = MODELS_INTERSECTION
    if args.n_models is not None:
        models = models[: args.n_models]
    df = load_wildbench(models)
    print_long_form_summary(df)
