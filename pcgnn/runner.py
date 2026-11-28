"""Execute one run and record everything it produces.

Run directory layout (``<output_root>/<experiment>/<run_id>/``)::

    config.yaml          resolved settings of this run (+ config hash)
    provenance.json      git commit, dirty files, packages, host, GPU
    git_diff.patch       uncommitted changes at launch time, if any
    status.json          pending | running | done | failed (+ timings, error)
    run.log              log of this run
    graph_meta.json      size of the preprocessed graph
    metrics_per_fold.csv fold, metric, scope, value
    costs_per_fold.csv   fold, duration, emissions, energy, inference time, epochs
    predictions.csv      fold, position, y_true, p_<class> for every test patient
    emissions.csv        raw CodeCarbon log

A run whose ``status.json`` says ``done`` with the same config hash is skipped.
An existing directory with a different config hash is never overwritten.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import provenance
from .config import RunConfig
from .folds import get_or_create_folds

log = logging.getLogger(__name__)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_dir(run: RunConfig, output_root: Path) -> Path:
    return output_root / run.experiment / run.run_id


def folds_path(run: RunConfig, folds_root: Path) -> Path:
    e = run.evaluation
    return folds_root / run.task / f"n{run.num_patients}_s{run.subsample}_{e.folds}fold_seed{e.split_seed}.json"


def read_status(path: Path) -> dict:
    f = path / "status.json"
    return json.loads(f.read_text()) if f.exists() else {"state": "pending"}


def write_status(path: Path, **fields) -> None:
    status = read_status(path)
    status.update(fields)
    tmp = path / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(path / "status.json")


def check_run_dir(run: RunConfig, out: Path) -> str:
    """'new', 'done', 'resume' or raise if the directory belongs to another config."""
    if not (out / "config.yaml").exists():
        return "new"
    saved = yaml.safe_load((out / "config.yaml").read_text())
    if saved.get("config_hash") != run.config_hash:
        raise RuntimeError(
            f"{out} exists with config hash {saved.get('config_hash')} != {run.config_hash}. "
            "Use a new experiment name or variant instead of overwriting results.")
    return "done" if read_status(out).get("state") == "done" else "resume"


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def execute(run: RunConfig, output_root: Path, cache_root: Path, folds_root: Path) -> Path:
    """Run every fold of ``run``; return its directory."""
    out = run_dir(run, output_root)
    state = check_run_dir(run, out)
    if state == "done":
        log.info("skip %s (done)", run.run_id)
        return out
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"config_hash": run.config_hash, **run.to_dict()},
                                                    sort_keys=False))
    provenance.write(out / "provenance.json")

    handler = logging.FileHandler(out / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("pcgnn").addHandler(handler)
    write_status(out, state="running", started=now(), error=None)
    try:
        _execute(run, out, cache_root, folds_root)
    except BaseException as err:
        write_status(out, state="failed", finished=now(), error=f"{type(err).__name__}: {err}",
                     traceback=traceback.format_exc())
        log.exception("run %s failed", run.run_id)
        raise
    finally:
        logging.getLogger("pcgnn").removeHandler(handler)
        handler.close()
    write_status(out, state="done", finished=now())
    return out


def _execute(run: RunConfig, out: Path, cache_root: Path, folds_root: Path) -> None:
    import torch

    from .costs import inference_time, measure
    from .graph import load_graph
    from .metrics import fold_metrics
    from .model import RGCNLit
    from .preprocess import ensure_preprocessed
    from .train import predict_proba, train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(run.seed)
    cache = ensure_preprocessed(run, cache_root, device)
    g = load_graph(run, cache)
    (out / "graph_meta.json").write_text(json.dumps(
        {**g.meta, "cache_dir": str(cache), "num_patients": run.num_patients}, indent=2))

    e = run.evaluation
    folds = get_or_create_folds(g.y, folds_path(run, folds_root), e.folds, e.val_fraction, e.split_seed,
                                meta={"task": run.task, "subsample": run.subsample,
                                      "labels_path": run.labels_path})

    data = g.data.to(device)
    patients = torch.as_tensor(g.patient_nodes, device=device)
    y = torch.as_tensor(g.y, device=device)
    binary = len(run.classes) == 2
    num_outputs = 1 if binary else len(run.classes)

    metrics, costs, preds = [], [], []
    for k, fold in enumerate(folds):
        tr, va, te = (torch.as_tensor(a, device=device) for a in (fold.train, fold.val, fold.test))
        gen = torch.Generator().manual_seed(run.seed * 1000 + k)
        torch.manual_seed(run.seed * 1000 + k)
        with measure(run.costs, out, f"fold_{k}") as cost:
            model = RGCNLit(data.num_nodes, data.num_relations, num_outputs, data.is_code.cpu(),
                            run.model, run.features, gen).to(device)
            result = train(model, data, patients[tr], y[tr], patients[va], y[va], run.train, binary)
            proba = predict_proba(model, data, patients[te], binary).cpu().numpy()
        t_inf = inference_time(model, data, run.costs.inference_repeats)

        y_test = g.y[fold.test]
        m = fold_metrics(y_test, proba, run.classes)
        m.insert(0, "fold", k)
        metrics.append(m)
        costs.append({"fold": k, "duration_s": cost.duration_s, "emissions_kg": cost.emissions_kg,
                      "energy_kwh": cost.energy_kwh, "inference_s": t_inf,
                      "inference_ms_per_patient": 1000 * t_inf / run.num_patients,
                      "epochs": result.epochs, "best_epoch": result.best_epoch,
                      "best_val_loss": result.best_val_loss,
                      "n_train": len(fold.train), "n_val": len(fold.val), "n_test": len(fold.test)})
        p = pd.DataFrame(proba, columns=[f"p_{c}" for c in run.classes])
        p.insert(0, "y_true", y_test)
        p.insert(0, "position", fold.test)
        p.insert(0, "fold", k)
        preds.append(p)
        macro_f1 = m.query("metric == 'f1' and scope == 'macro'")["value"].item()
        log.info("%s fold %d/%d: macro-F1 %.3f, %.1f s, best epoch %d",
                 run.run_id, k + 1, len(folds), macro_f1, cost.duration_s, result.best_epoch)
        del model

    pd.concat(metrics).to_csv(out / "metrics_per_fold.csv", index=False)
    pd.DataFrame(costs).to_csv(out / "costs_per_fold.csv", index=False)
    pd.concat(preds).to_csv(out / "predictions.csv", index=False)
