"""Classification metrics in long (tidy) format.

Every value carries an explicit ``scope`` (a class name, ``macro`` or
``weighted``), so per-class columns cannot be confused (finding F5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)


def fold_metrics(y_true: np.ndarray, proba: np.ndarray, classes: tuple[str, ...]) -> pd.DataFrame:
    y_pred = proba.argmax(axis=1)
    labels = list(range(len(classes)))
    rows = [("accuracy", "all", accuracy_score(y_true, y_pred))]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for k, name in enumerate(classes):
        rows += [("precision", name, p[k]), ("recall", name, r[k]), ("f1", name, f[k])]
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        rows += [("precision", avg, p), ("recall", avg, r), ("f1", avg, f)]
    if len(classes) == 2:
        rows += [("auc", "positive", roc_auc_score(y_true, proba[:, 1])),
                 ("auprc", "positive", average_precision_score(y_true, proba[:, 1]))]
    else:
        onehot = np.eye(len(classes))[y_true]
        for avg in ("macro", "weighted"):
            rows += [("auc", avg, roc_auc_score(y_true, proba, average=avg, multi_class="ovr")),
                     ("auprc", avg, average_precision_score(onehot, proba, average=avg))]
    return pd.DataFrame(rows, columns=["metric", "scope", "value"])


def summarize(per_fold: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and sample std over folds (and subsamples) for each metric/scope."""
    g = per_fold.groupby(by + ["metric", "scope"])["value"]
    out = g.agg(mean="mean", std=lambda v: v.std(ddof=1), n="count").reset_index()
    return out
