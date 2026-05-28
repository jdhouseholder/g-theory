from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent / "state" / "sweep_wildbench"
PER_CAP = ROOT / "per_cap"


def _cap_from_dirname(name: str) -> int:
    m = re.match(r"n_M_(\d+)", name)
    if not m:
        raise ValueError(f"unexpected dir name: {name}")
    return int(m.group(1))


def main() -> None:
    cap_dirs = sorted(
        [d for d in PER_CAP.iterdir() if d.is_dir() and d.name.startswith("n_M_")],
        key=lambda d: _cap_from_dirname(d.name),
    )
    print(f"[rebuild] {len(cap_dirs)} caps: {[d.name for d in cap_dirs]}")

    ds_rows: list[pd.DataFrame] = []
    var_rows: list[pd.DataFrame] = []
    log_rows: list[dict] = []
    for d in cap_dirs:
        cap = _cap_from_dirname(d.name)
        for fname in ("dstudy_latent.csv", "dstudy_probability.csv"):
            f = d / fname
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df.insert(0, "cap", cap)
            ds_rows.append(df)
        var_path = d / "variance_components.csv"
        if var_path.exists():
            v = pd.read_csv(var_path)
            v.insert(0, "cap", cap)
            var_rows.append(v)
        log_rows.append({"cap": cap, "dir": str(d), "from": "per_cap_reagg"})

    ds = pd.concat(ds_rows, ignore_index=True) if ds_rows else pd.DataFrame()
    var = pd.concat(var_rows, ignore_index=True) if var_rows else pd.DataFrame()

    ds.to_csv(ROOT / "aggregate_dstudy.csv", index=False)
    var.to_csv(ROOT / "aggregate_variance.csv", index=False)
    print(f"[rebuild] wrote aggregate_dstudy.csv ({len(ds)} rows)")
    print(f"[rebuild] wrote aggregate_variance.csv ({len(var)} rows)")

    if not ds.empty:
        from sweeps.common import headline_max, lmm_vs_glmm_gap

        h = headline_max(ds, ["cap"])
        h.to_csv(ROOT / "headline_max_design.csv", index=False)
        gap = lmm_vs_glmm_gap(h, ["cap"])
        gap.to_csv(ROOT / "headline_lmm_vs_glmm_gap.csv", index=False)
        print(f"[rebuild] wrote headline_max_design.csv ({len(h)} rows)")
        print(f"[rebuild] wrote headline_lmm_vs_glmm_gap.csv ({len(gap)} rows)")

    (ROOT / "sweep_log.txt").write_text(json.dumps(log_rows, indent=2))
    print(f"[rebuild] wrote sweep_log.txt (marker only, {len(log_rows)} rows)")


if __name__ == "__main__":
    main()
