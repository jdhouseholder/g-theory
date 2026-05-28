from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

from loaders import cached_url_get, print_long_form_summary

RUNS_DIR = Path(__file__).parent.parent / "datasets" / "helm_instruct" / "runs"

GCS_BUCKET = "crfm-helm-public"
GCS_PREFIX = "gzip/instruct/benchmark_output/runs/instruction_following/"
GCS_LIST_URL = (
    f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
    f"?prefix={GCS_PREFIX}&matchGlob=**/per_instance_stats.json&maxResults=500"
)
GCS_OBJECT_BASE = f"https://storage.googleapis.com/{GCS_BUCKET}/"

# 29 of 36 (scenario, criterion) pairs are modelable; the rest are degenerate:
# Keyword Feedback is a constant 0/missing indicator (entirely excluded below),
# and (vicuna, Harmlessness) saturates to two unique levels.
SCENARIOS = [
    "anthropic_hh_rlhf",
    "grammar",
    "koala",
    "open_assistant",
    "self_instruct",
    "vicuna",
]
CRITERIA_ALL = [
    "Helpfulness",
    "Understandability",
    "Completeness",
    "Conciseness",
    "Harmlessness",
]
SKIP = {("vicuna", "Harmlessness")}


def all_cells() -> list[tuple[str, str]]:
    return [(s, c) for s in SCENARIOS for c in CRITERIA_ALL if (s, c) not in SKIP]


def _gcs_path_to_local_name(obj_path: str) -> str:
    suffix = "/per_instance_stats.json"
    if not obj_path.startswith(GCS_PREFIX) or not obj_path.endswith(suffix):
        raise ValueError(f"Unexpected HELM object path: {obj_path}")
    run_id = obj_path[len(GCS_PREFIX) : -len(suffix)]
    run_id = (
        run_id.replace(":", "__").replace("=", "-").replace(",", "_").replace("/", "_")
    )
    return run_id + "__per_instance_stats.json"


def _list_remote_objects() -> list[str]:
    with urllib.request.urlopen(GCS_LIST_URL, timeout=30) as resp:
        data = json.load(resp)
    return [it["name"] for it in data.get("items", [])]


def _download(obj_path: str) -> Path:
    return cached_url_get(
        RUNS_DIR / _gcs_path_to_local_name(obj_path),
        GCS_OBJECT_BASE + obj_path,
    )


def fetch_all(verbose: bool = True) -> None:
    objs = _list_remote_objects()
    if verbose:
        print(f"[helm] {len(objs)} remote per_instance_stats files listed")
    pulled = 0
    for obj in objs:
        local = RUNS_DIR / _gcs_path_to_local_name(obj)
        if local.exists():
            continue
        _download(obj)
        pulled += 1
        if verbose:
            print(f"  pulled {local.name}")
    if verbose:
        n = sum(1 for _ in RUNS_DIR.glob("*_per_instance_stats.json"))
        print(f"[helm] {n} files cached in {RUNS_DIR} ({pulled} new this run)")


def parse_filename(
    name: str,
) -> tuple[str, str, str | None, str | None]:
    parts = name.split("__")
    bare_scenario = parts[0]
    middle = parts[1] if len(parts) >= 3 else ""
    m_eval = re.search(r"_evaluator-([A-Za-z0-9.\-]+)$", "_" + middle)
    judge = m_eval.group(1) if m_eval else None
    pre_eval = middle[: -len(f"evaluator-{judge}") - 1] if judge else middle
    m_model = re.search(r"_model-(.+)$", "_" + pre_eval)
    model = m_model.group(1) if m_model else None
    pre_model = pre_eval[: -len(f"model-{model}") - 1] if model else pre_eval
    full_scenario = (
        f"{bare_scenario}__{pre_model}" if pre_model else bare_scenario
    )
    return bare_scenario, full_scenario, model, judge


def load_helm(
    scenario: str, criterion: str, n_categories: int = 5
) -> pd.DataFrame:
    rows = []
    pattern = f"{scenario}__*per_instance_stats.json"
    files = sorted(RUNS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No HELM runs found for scenario={scenario!r} under {RUNS_DIR}"
        )
    for f in files:
        bare, _, model, judge = parse_filename(f.name)
        if bare != scenario or model is None or judge is None:
            continue
        data = json.load(open(f))
        for rec in data:
            iid = rec["instance_id"]
            for s in rec["stats"]:
                if s["name"]["name"] != criterion:
                    continue
                mean = s["mean"]
                k = int(round(float(mean)))
                k = max(1, min(n_categories, k))
                rows.append((model, iid, judge, k, float(mean)))
    df = pd.DataFrame(
        rows, columns=["model", "item", "judge", "score_ord", "score_raw"]
    )
    if df.empty:
        raise RuntimeError(
            f"Loaded zero rows for scenario={scenario!r} criterion={criterion!r}"
        )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="vicuna")
    parser.add_argument("--criterion", default="Helpfulness")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Download all 112 per_instance_stats.json files (idempotent).",
    )
    args = parser.parse_args()

    if args.prefetch:
        fetch_all()

    df = load_helm(args.scenario, args.criterion)
    print_long_form_summary(df)
