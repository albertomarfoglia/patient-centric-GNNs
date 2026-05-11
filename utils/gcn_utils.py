from collections import Counter

import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import os

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


# ------------------------ Device Selection ------------------------ #
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ------------------------ K-Fold Splits ------------------------ #
def k_fold(X, y, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=77)
    train_indices, val_indices, test_indices = [], [], []
    train_y, val_y, test_y = [], [], []

    for non_test_idx, test_idx in skf.split(X, y):
        test_indices.append(X[test_idx])
        train_idx, val_idx, _, _ = train_test_split(
            non_test_idx, y[non_test_idx], test_size=1 / 9, random_state=77
        )
        train_indices.append(X[train_idx])
        val_indices.append(X[val_idx])
        train_y.append(y[train_idx])
        val_y.append(y[val_idx])
        test_y.append(y[test_idx])

        # Print class distributions
        print("| Fold")
        print("|__ Train distribution:", Counter(y[train_idx]))
        print("|__ Val distribution:  ", Counter(y[val_idx]))
        print("|__ Test distribution: ", Counter(y[test_idx]))
        print("\n")

    return train_indices, val_indices, test_indices, train_y, val_y, test_y


# ------------------------ Metrics ------------------------ #
def compute_metrics(y_true, y_pred, y_prob, num_classes):
    """Compute accuracy, precision, recall, F1, and AUC."""

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # AUC
    if num_classes == 2:
        auc_class = roc_auc_score(y_true, y_prob[:, 1])
        auc_macro = auc_class
        auc_weighted = auc_class

        ap_class = average_precision_score(y_true, y_prob[:, 1])
        ap_macro = ap_class
        ap_weighted = ap_class
    else:
        # multi-class
        auc_class = roc_auc_score(y_true, y_prob, average=None, multi_class="ovr")
        auc_macro = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(
            y_true, y_prob, average="weighted", multi_class="ovr"
        )

        ap_class = average_precision_score(y_true, y_prob, average=None)
        ap_macro = average_precision_score(y_true, y_prob, average="macro")
        ap_weighted = average_precision_score(y_true, y_prob, average="weighted")

    # Precision, recall, F1
    precision_class, recall_class, fscore_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_weighted, recall_weighted, fscore_weighted, _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )

    metrics = {
        "accuracy": accuracy,
        "auc_class": auc_class,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "ap_class": ap_class,
        "ap_macro": ap_macro,
        "ap_weighted": ap_weighted,
        "precision_class": precision_class,
        "recall_class": recall_class,
        "fscore_class": fscore_class,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "fscore_macro": fscore_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "fscore_weighted": fscore_weighted,
    }

    return metrics


# ------------------------ Confusion Matrix ------------------------ #
def save_confusion_matrix(y_true, y_pred, result_path, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig = disp.plot().figure_
    fig.savefig(result_path, dpi=600)


def mean_std_metrics(
    metrics_mean: pd.DataFrame, metrics_std: pd.DataFrame, classes: list[str], digits=2
) -> pd.DataFrame:
    headers = classes + ["MACRO", "WEIGHTED"]

    metrics_mean = metrics_mean.reindex(headers)
    metrics_std = metrics_std.reindex(headers)

    def mean_std_str(mean, std, decimals=digits):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    f1_line = [
        mean_std_str(m, s)
        for m, s in zip(metrics_mean["F1SCORE"], metrics_std["F1SCORE"])
    ]

    f1_line.extend(
        [
            mean_std_str(
                metrics_mean.loc["WEIGHTED", "ACCURACY"],
                metrics_std.loc["WEIGHTED", "ACCURACY"],
            ),
            mean_std_str(
                metrics_mean.loc["WEIGHTED", "AUC"],
                metrics_std.loc["WEIGHTED", "AUC"]
            ),
            mean_std_str(
                metrics_mean.loc["WEIGHTED", "AP"],
                metrics_std.loc["WEIGHTED", "AP"]
            ),
        ]
    )

    return pd.DataFrame([f1_line], columns=(headers + ["Accuracy", "AUC", "AP"]))


def store_metrics(metrics: dict, classes: list[str], fold, out_path: str):
    if len(classes) == 2:
        metric_df = _binary_metrics(metrics, classes)
    else:
        metric_df = _multiclass_metrics(metrics, classes)

    metric_df.index.name = f"Fold_{fold}"
    metric_df.to_csv(out_path, mode="a")
    return metric_df


def _binary_metrics(metrics: dict, classes: list[str]):
    return pd.DataFrame(
        {
            "PRECISION": np.hstack(
                (
                    metrics["precision_class"],
                    metrics["precision_macro"],
                    metrics["precision_weighted"],
                )
            ),
            "RECALL": np.hstack(
                (
                    metrics["recall_class"],
                    metrics["recall_macro"],
                    metrics["recall_weighted"],
                )
            ),
            "F1SCORE": np.hstack(
                (
                    metrics["fscore_class"],
                    metrics["fscore_macro"],
                    metrics["fscore_weighted"],
                )
            ),
            "ACCURACY": np.hstack(
                (
                    np.zeros(len(classes)),  # per-class = 0
                    metrics["accuracy"],
                    metrics["accuracy"],
                )
            ),
            "AUC": np.hstack(
                (
                    np.repeat(
                        metrics["auc_macro"], len(classes)
                    ),  # same for both classes
                    metrics["auc_macro"],
                    metrics["auc_weighted"],
                )
            ),
            "AP": np.hstack(
                (
                    np.repeat(metrics["ap_macro"], len(classes)),
                    metrics["ap_macro"],
                    metrics["ap_weighted"],
                )
            ),
        },
        index=classes + ["MACRO", "WEIGHTED"],
    )


def _multiclass_metrics(metrics: dict, classes: list[str]):
    return pd.DataFrame(
        {
            "PRECISION": np.hstack(
                (
                    metrics["precision_class"],
                    metrics["precision_macro"],
                    metrics["precision_weighted"],
                )
            ),
            "RECALL": np.hstack(
                (
                    metrics["recall_class"],
                    metrics["recall_macro"],
                    metrics["recall_weighted"],
                )
            ),
            "F1SCORE": np.hstack(
                (
                    metrics["fscore_class"],
                    metrics["fscore_macro"],
                    metrics["fscore_weighted"],
                )
            ),
            "ACCURACY": np.hstack(
                (np.zeros(len(classes)), metrics["accuracy"], metrics["accuracy"])
            ),
            "AUC": np.hstack(
                (metrics["auc_class"], metrics["auc_macro"], metrics["auc_weighted"])
            ),
            "AP": np.hstack(
                (metrics["ap_class"], metrics["ap_macro"], metrics["ap_weighted"])
            ),
        },
        index=classes + ["MACRO", "WEIGHTED"],
    )


# ------------------------ Evaluation ------------------------ #
def evaluate_model(model, data, fold, result_dir, data_model, classes, time_opt):
    model.eval()

    if len(classes) > 2:
        with torch.no_grad():
            out = model(data)
            y_pred = out.argmax(dim=-1)
            y_prob = F.softmax(out, dim=-1)

            metrics = compute_metrics(
                data.test_y.cpu(),
                y_pred[data.test_idx].cpu(),
                y_prob[data.test_idx].cpu(),
                len(classes),
            )
    else:
        with torch.no_grad():
            out = model(data)
            y_prob = torch.sigmoid(out)
            y_pred = (y_prob > 0.5).long()

            y_prob_test = y_prob[data.test_idx].cpu()
            y_prob_2d = torch.stack([1 - y_prob_test, y_prob_test], dim=1)

            metrics = compute_metrics(
                data.test_y.cpu(),
                y_pred[data.test_idx].cpu(),
                y_prob_2d,
                len(classes),
            )

    metric_df = store_metrics(
        metrics,
        classes,
        fold,
        out_path=f"{result_dir}/metrics_{data_model}_{time_opt}_{data.num_patients}.csv",
    )

    # Save confusion matrix
    save_confusion_matrix(
        data.test_y.cpu(),
        y_pred[data.test_idx].cpu(),
        f"{result_dir}/cm/cm_{data_model}_{time_opt}_{data.num_patients}_{fold}.jpg",
        labels=classes,
    )

    y_folder = f"{result_dir}/{fold}"
    os.makedirs(y_folder, exist_ok=True)
    np.save(f"{y_folder}/y_true.npy", data.test_y.cpu().numpy())
    np.save(f"{y_folder}/y_index.npy", data.test_idx.cpu().numpy())
    np.save(f"{y_folder}/y_pred.npy", y_pred[data.test_idx].cpu().numpy())
    np.save(f"{y_folder}/y_prob.npy", y_prob[data.test_idx].cpu().numpy())

    return metric_df
