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

ROOT_URI_MAP = {
    "meds": "https://teamheka.github.io/meds-data/subject/",
    "sphn_pc": "http://nvasc.org/synth_patient_",
}

def load_data(num_patients: int, embed_dim: int, dcfg: LoaderConfig, inc_txt = False):
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
    # if vp := dcfg.text_values_path:
    #     data.txt_x = torch.Tensor(np.load(vp))

    num_x = torch.Tensor(np.load(dcfg.numeric_values_path)).view(-1, 1)

    data.num_mask = (~torch.isnan(num_x)).float()

    data.num_x = torch.nan_to_num(num_x, nan=0.0)

    if vp := inc_txt:
        data.txt_x = torch.tensor(np.load(vp)) # type: ignore
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
        val_loss = torch.nn.functional.nll_loss(out[data.valid_idx], data.valid_y) #, weight=class_weights)
        pred = out.argmax(dim=-1)

    return val_loss, pred


def train_model(model, data, lr, wd, max_epochs=2001, patience=30, binary=False):
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
            #TrainLoss=train_loss.item(),
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
):
    result_dir = loader.results_dir

    # Load data
    data, patients, y = load_data(
        num_patients,
        mcfg.embed_dim,
        loader,
        excfg.include_text
    )

    data.num_patients = num_patients

    # Save hyperparameters
    hyper_param = pd.DataFrame(
        {
            "DROPOUT": [mcfg.dropout],
            "LEARNING_RATE": [mcfg.lr],
            "WEIGHT_DECAY": [mcfg.weight_decay],
            "EMBED_DIM": [mcfg.embed_dim],
            "HIDDEN_DIM": [mcfg.hidden_dim],
        }
    )
    hyper_param.to_csv(
        f"{result_dir}/metrics_{excfg.time_option}_{num_patients}_hyperparams.csv",
        mode="a",
        index=False,
    )

    # K-Fold split
    (
        train_idx_list,
        val_idx_list,
        test_idx_list,
        train_y_list,
        val_y_list,
        test_y_list,
    ) = k_fold(np.asarray(patients), y, excfg.folds)

    # Get device and move data
    device = get_device()
    data = data.to(device)  # type: ignore

    all_metrics = []

    for fold, (train_idx, val_idx, test_idx, train_y, val_y, test_y) in enumerate(
        zip(
            train_idx_list,
            val_idx_list,
            test_idx_list,
            train_y_list,
            val_y_list,
            test_y_list,
        )
    ):
        # Move all indices and labels to the correct device
        data.train_idx = torch.Tensor(train_idx).long().to(device)
        data.valid_idx = torch.Tensor(val_idx).long().to(device)
        data.test_idx = torch.Tensor(test_idx).long().to(device)

        data.train_y = torch.Tensor(train_y).long().to(device) # CHANGED
        data.valid_y = torch.Tensor(val_y).long().to(device)
        data.test_y = torch.Tensor(test_y).long().to(device)

        os.makedirs(f"{result_dir}/{fold}", exist_ok=True)

        # Initialize and train model
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
            binary=(loader.num_classes == 2)
        )

        # Evaluate
        metric = evaluate_model(
            model,
            data,
            fold,
            result_dir,
            loader.data_mode,
            loader.classes,
            excfg.time_option,
        )
        all_metrics.append(metric)

    # Aggregate metrics
    panel = pd.concat(all_metrics)
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
