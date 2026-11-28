"""Command-line interface.

    python -m pcgnn list      SPEC [--filter REGEX]
    python -m pcgnn status    SPEC [--filter REGEX]
    python -m pcgnn run       SPEC [--gpus 0,1,2,3] [--filter REGEX] [--retry-failed]
    python -m pcgnn folds     SPEC                 # write the shared fold files only
    python -m pcgnn summarize SPEC                 # aggregate finished runs
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from .config import ExperimentSpec, RunConfig
from .logging_setup import setup_logging
from .runner import check_run_dir, folds_path, read_status, run_dir

log = logging.getLogger("pcgnn")
FOLDS_ROOT = Path("folds")


def _select(spec: ExperimentSpec, pattern: str | None) -> list[RunConfig]:
    runs = spec.expand()
    if pattern:
        rx = re.compile(pattern)
        runs = [r for r in runs if rx.search(r.run_id)]
    return runs


def cmd_list(spec, runs, args):
    for r in runs:
        print(r.run_id)
    print(f"{len(runs)} runs", file=sys.stderr)


def cmd_status(spec, runs, args):
    counts: dict[str, int] = {}
    for r in runs:
        st = read_status(run_dir(r, Path(spec.output_root)))
        counts[st["state"]] = counts.get(st["state"], 0) + 1
        if st["state"] in ("failed", "running") or args.verbose:
            print(f"{st['state']:8s} {r.run_id} {st.get('error') or ''}")
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def cmd_folds(spec, runs, args):
    import joblib

    from .folds import get_or_create_folds

    seen = set()
    for r in runs:
        path = folds_path(r, FOLDS_ROOT)
        if path in seen:
            continue
        seen.add(path)
        y = joblib.load(r.labels_path)
        e = r.evaluation
        get_or_create_folds(y, path, e.folds, e.val_fraction, e.split_seed,
                            meta={"task": r.task, "subsample": r.subsample, "labels_path": r.labels_path})
        print(path)


def cmd_run(spec, runs, args):
    from .scheduler import run_parallel

    out_root, cache_root = Path(spec.output_root), Path(spec.cache_root)
    todo = []
    for r in runs:
        d = run_dir(r, out_root)
        state = check_run_dir(r, d)  # raises on a config clash
        if state == "done":
            continue
        if read_status(d).get("state") == "failed" and not args.retry_failed:
            log.info("skip failed run %s (use --retry-failed)", r.run_id)
            continue
        if read_status(d).get("state") == "running":
            log.warning("%s was left 'running' (interrupted?); running it again", r.run_id)
        todo.append(r)
    log.info("%d of %d runs to execute on GPUs %s", len(todo), len(runs), args.gpus)
    if args.dry_run:
        for r in todo:
            print(r.run_id)
        return
    summary = run_parallel(todo, args.gpus.split(","), out_root, cache_root, FOLDS_ROOT)
    log.info("finished: %d done, %d failed", len(summary["done"]), len(summary["failed"]))
    if summary["failed"]:
        sys.exit(1)


def cmd_summarize(spec, runs, args):
    from .metrics import summarize

    out_root = Path(spec.output_root)
    m, c = [], []
    keys = ["variant", "task", "schema", "pruning"]
    for r in runs:
        d = run_dir(r, out_root)
        if read_status(d).get("state") != "done":
            continue
        ident = {"variant": r.variant, "task": r.task, "schema": r.schema, "pruning": r.pruning,
                 "subsample": r.subsample}
        m.append(pd.read_csv(d / "metrics_per_fold.csv").assign(**ident))
        c.append(pd.read_csv(d / "costs_per_fold.csv").assign(**ident))
    if not m:
        print("no finished runs")
        return
    dest = out_root / spec.name / "summary"
    dest.mkdir(parents=True, exist_ok=True)
    metrics, costs = pd.concat(m), pd.concat(c)
    metrics.to_csv(dest / "metrics_all_folds.csv", index=False)
    costs.to_csv(dest / "costs_all_folds.csv", index=False)
    summarize(metrics, keys).to_csv(dest / "metrics_summary.csv", index=False)
    cost_cols = ["duration_s", "emissions_kg", "energy_kwh", "inference_s", "inference_ms_per_patient", "epochs"]
    costs.groupby(keys)[cost_cols].agg(["mean", "std", "count"]).to_csv(dest / "costs_summary.csv")
    print(f"written to {dest} ({metrics[keys + ['subsample']].drop_duplicates().shape[0]} runs)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="pcgnn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "status", "run", "folds", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("spec", type=Path)
        s.add_argument("--filter", help="regular expression on run ids")
        if name == "status":
            s.add_argument("-v", "--verbose", action="store_true")
        if name == "run":
            s.add_argument("--gpus", default="0", help="comma-separated physical GPU ids")
            s.add_argument("--retry-failed", action="store_true")
            s.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    setup_logging()
    spec = ExperimentSpec.load(args.spec)
    runs = _select(spec, args.filter)
    {"list": cmd_list, "status": cmd_status, "run": cmd_run, "folds": cmd_folds,
     "summarize": cmd_summarize}[args.cmd](spec, runs, args)


if __name__ == "__main__":
    main()
