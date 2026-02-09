from torch_geometric.logging import log
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, ConfusionMatrixDisplay
)
import pandas as pd

# ------------------------ Device Selection ------------------------ #
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# ------------------------ K-Fold Splits ------------------------ #
def k_fold(X, y, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=77)
    train_indices, val_indices, test_indices = [], [], []
    train_y, val_y, test_y = [], [], []

    for non_test_idx, test_idx in skf.split(X, y):
        test_indices.append(X[test_idx])
        train_idx, val_idx, _, _ = train_test_split(
            non_test_idx, y[non_test_idx], test_size=1/9, random_state=77
        )
        train_indices.append(X[train_idx])
        val_indices.append(X[val_idx])
        train_y.append(y[train_idx])
        val_y.append(y[val_idx])
        test_y.append(y[test_idx])

    return train_indices, val_indices, test_indices, train_y, val_y, test_y


# ------------------------ Metrics ------------------------ #
def compute_metrics(y_true, y_pred, y_prob):
    """Compute accuracy, precision, recall, F1, and AUC."""
    accuracy = accuracy_score(y_true, y_pred)
    auc_class = roc_auc_score(y_true, y_prob, average=None, multi_class='ovr')
    auc_macro = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
    auc_weighted = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
    precision_class, recall_class, fscore_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, fscore_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'auc_class': auc_class,
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted,
        'precision_class': precision_class,
        'recall_class': recall_class,
        'fscore_class': fscore_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'fscore_macro': fscore_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'fscore_weighted': fscore_weighted
    }

    return metrics

# ------------------------ Confusion Matrix ------------------------ #
def save_confusion_matrix(y_true, y_pred, result_path, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig = disp.plot().figure_
    fig.savefig(result_path, dpi=600)

def mean_std_metrics(metrics_mean: pd.DataFrame, metrics_std: pd.DataFrame, digits=2) -> pd.DataFrame:
    headers = ["B2H", "REHAB", "DEATH", "MACRO", "WEIGHTED"]

    metrics_mean = metrics_mean.reindex(headers)
    metrics_std  = metrics_std.reindex(headers)

    def mean_std_str(mean, std, decimals=digits):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    f1_line = [
        mean_std_str(m, s)
        for m, s in zip(metrics_mean["F1SCORE"], metrics_std["F1SCORE"])
    ]

    f1_line.extend([
        mean_std_str(metrics_mean.loc["WEIGHTED", "ACCURACY"],
                    metrics_std.loc["WEIGHTED", "ACCURACY"]),
        mean_std_str(metrics_mean.loc["WEIGHTED", "AUC"],
                    metrics_std.loc["WEIGHTED", "AUC"]),
    ])

    return pd.DataFrame([f1_line], columns=(headers + ["Accuracy", "AUC"]))