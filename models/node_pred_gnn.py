import math
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.logging import log
import os
from configs.experiment import ExperimentConfig
from configs.loader import LoaderConfig
from configs.model import ModelConfig
from utils.gcn_utils import (
    get_device,
    k_fold,
    mean_std_metrics,
    evaluate_model,
)
from codecarbon import EmissionsTracker
import time

from utils.metrics import SubsampleResult

ROOT_URI_MAP = {
    "meds": "https://teamheka.github.io/meds-data/subject/",
    "sphn_pc": "http://nvasc.org/synth_patient_",
}


def load_data(num_patients: int, embed_dim: int, dcfg: LoaderConfig, inc_txt=False):
    entity_df = pd.read_csv(
        dcfg.entities_path,
        sep="\t",
        header=None,
    )
    entity_dict = dict(zip(entity_df[1], entity_df[0]))

    patients = [
        entity_dict[f"{ROOT_URI_MAP[dcfg.data_mode]}{i}"] for i in range(num_patients)
    ]
    y = np.asarray(joblib.load(dcfg.outcomes_path))

    triples = pd.read_csv(
        dcfg.triples_path,
        sep="\t",
        header=None,
    )

    triples_inv = triples[[2, 1, 0]]
    triples_inv.columns = [0, 1, 2]
    triples = triples_inv

    edge_index = torch.vstack(
        (torch.Tensor(triples[0]).long(), torch.Tensor(triples[2]).long())
    )
    edge_type = torch.Tensor(triples[1]).long()
    num_nodes = len(entity_dict)

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_classes=dcfg.num_classes,
    )

    data.num_relations = edge_type.max().item() + 1

    data.x = torch.nn.init.xavier_uniform_(
        tensor=Parameter(torch.empty(num_nodes, embed_dim)), gain=math.sqrt(2.0)
    )

    num_x = torch.Tensor(np.load(dcfg.numeric_values_path)).view(-1, 1)

    data.num_mask = (~torch.isnan(num_x)).float()

    data.num_x = torch.nan_to_num(num_x, nan=0.0)

    if inc_txt:
        data.txt_x = torch.tensor(np.load(dcfg.text_values_path))  # type: ignore
        data.txt_mask = (data.txt_x.abs().sum(dim=1) != 0).float()

    return data, patients, y


def _compute_binary_classification(model, data, optimizer):
    out = model(data)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_loss = criterion(out[data.train_idx], data.train_y.float())

    train_loss.backward()
    optimizer.step()
    model.eval()

    with torch.no_grad():
        out = model(data)

        val_loss = criterion(out[data.valid_idx], data.valid_y.float())
        probs = torch.sigmoid(out)
        pred = (probs > 0.5).long()

    return val_loss, pred


def _compute_multi_classification(model, data, optimizer):
    out = model(data)
    train_loss = torch.nn.functional.nll_loss(out[data.train_idx], data.train_y)
    train_loss.backward()
    optimizer.step()
    model.eval()

    # from sklearn.utils.class_weight import compute_class_weight

    # classes = np.unique(data.train_y.cpu().numpy())

    # weights = compute_class_weight(
    #     class_weight="balanced",
    #     classes=classes,
    #     y=data.train_y.cpu().numpy()
    # )

    # class_weights = torch.tensor(weights, dtype=torch.float).to(get_device())

    with torch.no_grad():
        out = model(data)
        val_loss = torch.nn.functional.nll_loss(
            out[data.valid_idx], data.valid_y
        )  # , weight=class_weights)
        pred = out.argmax(dim=-1)

    return val_loss, pred


def train_model(model, data, lr, wd, max_epochs=2001, patience=50, binary=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float("inf")
    min_delta = 1e-4
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ---- training ----
        model.train()
        optimizer.zero_grad()

        if binary:
            val_loss, pred = _compute_binary_classification(model, data, optimizer)
        else:
            val_loss, pred = _compute_multi_classification(model, data, optimizer)

        train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
        val_acc = float((pred[data.valid_idx] == data.valid_y).float().mean())
        test_acc = float((pred[data.test_idx] == data.test_y).float().mean())

        # ---- early stopping logic ----
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), data.model_path)
        else:
            epochs_no_improve += 1

        log(
            Epoch=epoch,
            # TrainLoss=train_loss.item(),
            ValLoss=val_loss.item(),
            Train=train_acc,
            Val=val_acc,
            Test=test_acc,
        )

        if epochs_no_improve >= patience:
            log(EarlyStop=f"Stopped at epoch {epoch}")
            break

    model.load_state_dict(torch.load(data.model_path, weights_only=True))
    return model


def run_gnn(
    num_patients,
    mcfg: ModelConfig,
    loader: LoaderConfig,
    excfg: ExperimentConfig,
) -> SubsampleResult:

    result_dir = loader.results_dir

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    data, patients, y = load_data(
        num_patients,
        mcfg.embed_dim,
        loader,
        excfg.include_text,
    )

    data.num_patients = num_patients

    # --------------------------------------------------------
    # K-fold split
    # --------------------------------------------------------
    (
        train_idx_list,
        val_idx_list,
        test_idx_list,
        train_y_list,
        val_y_list,
        test_y_list,
    ) = k_fold(np.asarray(patients), y, excfg.folds)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = get_device()
    data = data.to(device) # type: ignore

    fold_metrics = []
    fold_costs = []

    # --------------------------------------------------------
    # Train/evaluate folds
    # --------------------------------------------------------
    for fold, (
        train_idx,
        val_idx,
        test_idx,
        train_y,
        val_y,
        test_y,
    ) in enumerate(
        zip(
            train_idx_list,
            val_idx_list,
            test_idx_list,
            train_y_list,
            val_y_list,
            test_y_list,
        )
    ):
        tracker = EmissionsTracker(
            project_name=f"fold_{fold}",
            output_dir=str(result_dir),
            measure_power_secs=1,
        )

        tracker.start()
        start = time.perf_counter()

        # Indices and labels
        data.train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        data.valid_idx = torch.as_tensor(val_idx, dtype=torch.long, device=device)
        data.test_idx = torch.as_tensor(test_idx, dtype=torch.long, device=device)

        data.train_y = torch.as_tensor(train_y, dtype=torch.long, device=device)
        data.valid_y = torch.as_tensor(val_y, dtype=torch.long, device=device)
        data.test_y = torch.as_tensor(test_y, dtype=torch.long, device=device)

        os.makedirs(f"{result_dir}/{fold}", exist_ok=True)

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------
        model = excfg.model_type(
            embed_dim=mcfg.embed_dim,
            hidden_dim=mcfg.hidden_dim,
            num_relations=data.num_relations,
            dropout=mcfg.dropout,
            num_classes=loader.num_classes,
            include_text_features=excfg.include_text,
        ).to(device)

        data.model_path = f"{result_dir}/{fold}/model_weights.pth"

        model = train_model(
            model,
            data,
            lr=mcfg.lr,
            wd=mcfg.weight_decay,
            binary=(loader.num_classes == 2),
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------
        metrics = evaluate_model(
            model,
            data,
            fold,
            result_dir,
            loader.data_mode,
            loader.classes,
            excfg.time_option,
        )

        fold_metrics.append(metrics)

        # ----------------------------------------------------
        # Computational cost
        # ----------------------------------------------------
        duration = time.perf_counter() - start
        emissions = tracker.stop()

        fold_costs.append(
            {
                "fold": fold,
                "duration": duration,
                "emissions": emissions,
            }
        )

        print(f"Fold {fold}: duration={duration:.2f}s, emissions={emissions:.4f}")

    panel = pd.concat(fold_metrics)
    metrics_mean = panel.groupby(level=0).mean()
    metrics_mean.index.name = "MEAN"
    metrics_std = panel.groupby(level=0).std()
    metrics_std.index.name = "STD"
    mean_std_metrics(metrics_mean, metrics_std, loader.classes).to_csv(
        f"{result_dir}/metrics_{excfg.time_option}_{num_patients}_mean_std.csv",
        sep="\t",
        index=False,
        mode="a",
    )
    metrics_mean.to_csv(
        f"{result_dir}/metrics_{excfg.time_option}_{num_patients}.csv", mode="a"
    )
    metrics_std.to_csv(
        f"{result_dir}/metrics_{excfg.time_option}_{num_patients}.csv", mode="a"
    )

    # --------------------------------------------------------
    # Build result object
    # --------------------------------------------------------
    return SubsampleResult(
        fold_metrics=pd.concat(fold_metrics),
        fold_costs=pd.DataFrame(fold_costs),
    )
