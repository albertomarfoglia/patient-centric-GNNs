"""Cross-validation fold assignment shared by the graph and tabular models.

Folds are computed once per (task, subsample) from the labels and saved as
JSON. Every model family reads the same file, so patient assignments are
identical by construction (finding F7: the first submission's RGCN runs used
8 folds while the tabular runs used 10).

Positions refer to the order of the label vector, i.e. subject ``i`` of the
subsample is position ``i``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_folds(y: np.ndarray, n_folds: int, val_fraction: float, seed: int) -> list[Fold]:
    """Stratified K-fold test sets; validation drawn from the remaining patients.

    Identical to ``utils.gcn_utils.k_fold`` of the first submission (same
    estimator, shuffling and seeds), so legacy runs can be reproduced.
    """
    y = np.asarray(y)
    positions = np.arange(len(y))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    folds = []
    for non_test, test in skf.split(positions, y):
        train, val = train_test_split(non_test, test_size=val_fraction, random_state=seed)
        folds.append(Fold(train=np.sort(train), val=np.sort(val), test=np.sort(test)))
    _check_partition(folds, len(y))
    return folds


def _check_partition(folds: list[Fold], n: int) -> None:
    tests = np.concatenate([f.test for f in folds])
    if len(tests) != n or len(np.unique(tests)) != n:
        raise AssertionError("test folds do not partition the patients")
    for k, f in enumerate(folds):
        if set(f.train) & set(f.val) or set(f.train) & set(f.test) or set(f.val) & set(f.test):
            raise AssertionError(f"fold {k}: overlapping train/val/test")


def save_folds(folds: list[Fold], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "folds": [{"train": f.train.tolist(), "val": f.val.tolist(), "test": f.test.tolist()} for f in folds],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def load_folds(path: Path) -> list[Fold]:
    payload = json.loads(Path(path).read_text())
    return [Fold(*(np.asarray(f[k], dtype=int) for k in ("train", "val", "test"))) for f in payload["folds"]]


def get_or_create_folds(y: np.ndarray, path: Path, n_folds: int, val_fraction: float,
                        seed: int, meta: dict) -> list[Fold]:
    """Load the saved folds, or create and save them. A saved file whose
    settings differ from the requested ones is an error, never overwritten."""
    if path.exists():
        saved = json.loads(path.read_text())["meta"]
        expected = {"n_folds": n_folds, "val_fraction": val_fraction, "seed": seed, "n": int(len(y))}
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{path} was created with different settings: {saved} vs {expected}")
        return load_folds(path)
    folds = make_folds(y, n_folds, val_fraction, seed)
    save_folds(folds, path, {**meta, "n_folds": n_folds, "val_fraction": val_fraction,
                             "seed": seed, "n": int(len(y))})
    return folds
