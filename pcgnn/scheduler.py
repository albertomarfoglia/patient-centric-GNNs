"""Run many runs on several GPUs: one worker process per GPU.

Each worker is a fresh (spawned) process restricted to its GPU through
CUDA_VISIBLE_DEVICES, and pulls whole runs (all folds of one subsample) from a
shared queue, so GPUs stay busy until the queue is empty. A failed run is
recorded in its status.json and does not stop the other runs.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as queue_mod
from pathlib import Path

from .config import RunConfig

log = logging.getLogger(__name__)


def _worker(gpu: str, jobs, results, output_root: str, cache_root: str, folds_root: str) -> None:
    # Must happen before CUDA is initialised in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from .logging_setup import setup_logging
    from .runner import execute

    setup_logging(prefix=f"[gpu{gpu}] ")
    while True:
        try:
            run: RunConfig = jobs.get_nowait()
        except queue_mod.Empty:
            return
        try:
            execute(run, Path(output_root), Path(cache_root), Path(folds_root))
            results.put((run.run_id, "done", None))
        except Exception as err:  # recorded in status.json by execute()
            results.put((run.run_id, "failed", f"{type(err).__name__}: {err}"))


def run_parallel(runs: list[RunConfig], gpus: list[str], output_root: Path, cache_root: Path,
                 folds_root: Path) -> dict[str, list[str]]:
    if not runs:
        return {"done": [], "failed": []}
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    for r in runs:
        jobs.put(r)
    procs = [ctx.Process(target=_worker, name=f"gpu{g}",
                         args=(g, jobs, results, str(output_root), str(cache_root), str(folds_root)))
             for g in gpus[: len(runs)]]
    for p in procs:
        p.start()
    summary: dict[str, list[str]] = {"done": [], "failed": []}
    for i in range(len(runs)):
        while True:
            try:
                run_id, state, err = results.get(timeout=30)
                break
            except queue_mod.Empty:
                if not any(p.is_alive() for p in procs):
                    log.error("all workers exited with %d runs unaccounted for", len(runs) - i)
                    return summary
        summary[state].append(run_id)
        log.info("[%d/%d] %s %s%s", i + 1, len(runs), state, run_id, f" ({err})" if err else "")
    for p in procs:
        p.join()
    return summary
