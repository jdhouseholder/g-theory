from __future__ import annotations

import argparse
import dataclasses
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent
PY = sys.executable


@dataclasses.dataclass
class Step:
    name: str
    kind: str
    script: str
    args: list[str] = dataclasses.field(default_factory=list)
    smoke_args: list[str] | None = None

    def resolved(self, smoke: bool) -> list[str]:
        a = self.smoke_args if (smoke and self.smoke_args is not None) else self.args
        return [PY, self.script, *a]


SMOKE_FIT_ARGS = ["--warmup", "200", "--samples", "200", "--chains", "1"]

STEPS: list[Step] = [
    Step(
        "sweep_helm",
        "fit",
        "sweeps/sweep_helm.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--limit", "1"],
    ),
    Step(
        "sweep_helm_het",
        "fit",
        "sweeps/sweep_helm_het.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--limit", "1"],
    ),
    Step(
        "sweep_biggen",
        "fit",
        "sweeps/sweep_biggen.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--limit", "1"],
    ),
    Step(
        "sweep_biggen_het",
        "fit",
        "sweeps/sweep_biggen_het.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--only", "reasoning"],
    ),
    Step(
        "sweep_wildbench",
        "fit",
        "sweeps/sweep_wildbench.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--only", "4"],
    ),
    Step(
        "sweep_wildbench_het",
        "fit",
        "sweeps/sweep_wildbench_het.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--only", "4"],
    ),
    Step(
        "validate_misspecified",
        "fit",
        "validations/validate_misspecified.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--reps", "1"],
    ),
    Step(
        "validate_het_on_misspec",
        "fit",
        "validations/validate_het_on_misspec.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--reps", "1"],
    ),
    Step(
        "validate_n_M_sweep",
        "fit",
        "validations/validate_n_M_sweep.py",
        smoke_args=[*SMOKE_FIT_ARGS, "--reps", "1"],
    ),
    Step("fig_recommendations", "figure", "figures/figure_recommendations.py"),
    Step("fig_heteroscedasticity", "figure", "figures/figure_heteroscedasticity.py"),
]

DSTUDY_SCRIPT = "dstudy_recommendations.py"


def _select(
    steps: list[Step],
    only: list[str],
    skip: list[str],
    figures_only: bool,
    fits_only: bool,
) -> list[Step]:
    names = [s.name for s in steps]
    for bad in (set(only) | set(skip)) - set(names):
        raise SystemExit(f"unknown step: {bad}\nknown: {names}")
    out = []
    for s in steps:
        if only and s.name not in only:
            continue
        if s.name in skip:
            continue
        if figures_only and s.kind != "figure":
            continue
        if fits_only and s.kind != "fit":
            continue
        out.append(s)
    return out


def _run_step(step: Step, smoke: bool, log_dir: Path) -> tuple[bool, float]:
    cmd = step.resolved(smoke)
    log_path = log_dir / f"{step.name}.log"
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n>>> [{step.kind}] {step.name}")
    print(f"    {pretty}")
    print(f"    log: {log_path.relative_to(REPO)}")
    start = time.time()
    with log_path.open("w") as f:
        f.write(f"# {pretty}\n# started {datetime.now().isoformat()}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
    elapsed = time.time() - start
    ok = rc == 0
    print(f"    -> {'ok' if ok else f'FAILED rc={rc}'}  ({elapsed:.1f}s)")
    return ok, elapsed


def cmd_list(args: argparse.Namespace) -> int:
    selected = _select(STEPS, [], [], False, False)
    print(f"{'STEP':<28}  KIND     COMMAND")
    for s in selected:
        cmd = " ".join(s.resolved(args.smoke)[1:])
        print(f"{s.name:<28}  {s.kind:<7}  {cmd}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.figures_only and args.fits_only:
        raise SystemExit("--figures-only and --fits-only are mutually exclusive")

    selected = _select(STEPS, args.only, args.skip, args.figures_only, args.fits_only)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO / "state" / "logs" / f"repro_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"# main.py run - {len(selected)} step(s), smoke={args.smoke}")
    print(f"# logs: {log_dir.relative_to(REPO)}")

    results: list[tuple[Step, bool, float]] = []
    for step in selected:
        ok, elapsed = _run_step(step, args.smoke, log_dir)
        results.append((step, ok, elapsed))
        if not ok and args.stop_on_failure:
            break

    print("\n# summary")
    total = sum(r[2] for r in results)
    for step, ok, elapsed in results:
        print(f"  {'ok ' if ok else 'FAIL'}  {step.name:<28}  {elapsed:7.1f}s")
    failed = [s.name for s, ok, _ in results if not ok]
    print(f"# total {total:.1f}s  |  {len(results) - len(failed)}/{len(results)} ok")
    return 0 if not failed else 1


def cmd_dstudy(args: argparse.Namespace) -> int:
    cmd = [PY, DSTUDY_SCRIPT]
    print(f">>> {shlex.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{run,list,dstudy}")

    p_run = sub.add_parser(
        "run", help="Run the reproduction pipeline (fits + figures)."
    )
    p_run.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny warmup/samples + limit=1; verifies plumbing in minutes.",
    )
    p_run.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip all fits; just regenerate PNGs from cached state/.",
    )
    p_run.add_argument(
        "--fits-only", action="store_true", help="Run fits but not figures."
    )
    p_run.add_argument(
        "--only", nargs="*", default=[], help="Run only these step names."
    )
    p_run.add_argument("--skip", nargs="*", default=[], help="Skip these step names.")
    p_run.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Halt on first failure (default: continue and report at end).",
    )
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="Print the step plan and exit.")
    p_list.add_argument(
        "--smoke", action="store_true", help="Show the smoke-mode command line."
    )
    p_list.set_defaults(func=cmd_list)

    p_dstudy = sub.add_parser(
        "dstudy",
        help="Aggregate D-study recommendations across all sweeps (post-hoc).",
    )
    p_dstudy.set_defaults(func=cmd_dstudy)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
