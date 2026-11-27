"""Computational cost: duration, CodeCarbon emissions and inference time.

Duration and emissions cover the same block (model construction, training and
test-set prediction of one fold), as in the first submission. Inference time is
measured afterwards, outside the CodeCarbon block.

When several runs share a machine, use ``tracking_mode='process'``: CodeCarbon
then attributes CPU and RAM to this process and, through CUDA_VISIBLE_DEVICES,
tracks only the GPU assigned to this worker.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import CostConfig

log = logging.getLogger(__name__)


@dataclass
class Cost:
    duration_s: float = float("nan")
    emissions_kg: float = float("nan")
    energy_kwh: float = float("nan")


@contextmanager
def measure(cfg: CostConfig, out_dir: Path, name: str):
    """Yield a ``Cost`` that is filled in when the block exits."""
    cost = Cost()
    tracker = None
    if cfg.track_emissions:
        from codecarbon import OfflineEmissionsTracker

        tracker = OfflineEmissionsTracker(
            project_name=name, output_dir=str(out_dir), measure_power_secs=cfg.measure_power_secs,
            tracking_mode=cfg.tracking_mode, log_level="error", allow_multiple_runs=True,
            country_iso_code=cfg.country_iso_code,
        )
        tracker.start()
    start = time.perf_counter()
    try:
        yield cost
    finally:
        cost.duration_s = time.perf_counter() - start
        if tracker is not None:
            emissions = tracker.stop()
            cost.emissions_kg = float(emissions) if emissions is not None else float("nan")
            data = getattr(tracker, "final_emissions_data", None)
            if data is not None:
                cost.energy_kwh = float(data.energy_consumed)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def inference_time(model, data, repeats: int) -> float:
    """Mean seconds of one full-graph forward pass in eval mode, which scores
    every patient node; one warm-up pass, GPU-synchronised."""
    model.eval()
    model(data)
    _sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(data)
        _sync()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))
