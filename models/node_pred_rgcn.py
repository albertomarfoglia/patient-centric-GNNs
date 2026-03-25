import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear, PReLU
from torch_geometric.data import Data
from torch_geometric.logging import log
from torch_geometric.nn import RGCNConv

from configs.loader import LoaderConfig

from configs.model import ModelConfig
from utils.gcn_utils import (
    get_device,
    k_fold,
    mean_std_metrics,
    evaluate_model,
)


# ------------------------ RGCN Model ------------------------ #
class RGCNNet(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_relations: int,
        dropout: float,
        num_classes: int,
        include_text_features: bool,
    ):
        super().__init__()

        self.include_text_features = include_text_features
        self.dropout = dropout

        # ----- Feature projections -----
        self.numeric_projection = Linear(1, embed_dim)

        if self.include_text_features:
            self.text_projection = Linear(384, embed_dim)

        self.input_activation = PReLU(embed_dim)

        # ----- RGCN layers -----
        self.conv1 = RGCNConv(embed_dim, hidden_dim, num_relations, num_bases=8)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=8)
        self.conv3 = RGCNConv(hidden_dim, num_classes, num_relations, num_bases=8)

        self.act1 = PReLU(hidden_dim)
        self.act2 = PReLU(hidden_dim)

    def forward(self, data):
        # ----- Initial feature fusion -----
        node_features = self.numeric_projection(data.num_x)
        node_features = self.input_activation(node_features)

        if self.include_text_features:
            text_features = self.text_projection(data.txt_x)
            text_features = self.input_activation(text_features)
            node_features = node_features + text_features

        node_features = node_features + data.x

        # ----- Graph convolutions -----
        node_features = self.conv1(node_features, data.edge_index, data.edge_type)
        node_features = self.act1(node_features)
        node_features = F.dropout(node_features, p=self.dropout, training=self.training)

        node_features = self.conv2(node_features, data.edge_index, data.edge_type)
        node_features = self.act2(node_features)
        node_features = F.dropout(node_features, p=self.dropout, training=self.training)

        node_features = self.conv3(node_features, data.edge_index, data.edge_type)

        return F.log_softmax(node_features, dim=1)


ROOT_URI_MAP = {
    "meds": "https://teamheka.github.io/meds-data/subject/",
    "sphn_pc": "http://nvasc.org/synth_patient_",
}


# ------------------------ Data Loading ------------------------ #
def load_data(num_patients: int, embed_dim: int, dcfg: LoaderConfig):
    entity_df = pd.read_csv(
        dcfg.entities_path,
        # f"processed_data/{data_model}_{timeOpt}_entities_{num_patients}_{idx}.tsv",
        sep="\t",
        header=None,
    )
    # entity_dict: dict = entity_df.set_index(entity_df[1]).to_dict()[0]
    entity_dict = dict(zip(entity_df[1], entity_df[0]))

    patients = [
        entity_dict[f"{ROOT_URI_MAP[dcfg.data_mode]}{i}"] for i in range(num_patients)
    ]
    y = np.asarray(
        joblib.load(dcfg.outcomes_path)
        # joblib.load(f"data/outcomes_{data_model}_{timeOpt}_{num_patients}_{idx}.joblib")
        # joblib.load(f"/home/ubuntu/workspace/meds-to-owl-examples/exports/inhospital_mortality/labels/outcomes_{data_model}_{timeOpt}_{num_patients}_{idx}.joblib")
    )

    triples = pd.read_csv(
        dcfg.triples_path,
        # f"processed_data/{data_model}_{timeOpt}_triples_{num_patients}_{idx}.tsv",
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

    data.x = torch.nn.init.xavier_uniform_(
        tensor=Parameter(torch.empty(num_nodes, embed_dim)), gain=math.sqrt(2.0)
    )
    if vp := dcfg.text_values_path:
        data.txt_x = torch.Tensor(
            np.load(vp)
            # np.load(f"processed_data/{data_model}_{timeOpt}_text_{num_patients}_{idx}.npy")
        )
    data.num_x = torch.Tensor(
        np.load(dcfg.numeric_values_path)
        # np.load(f"processed_data/{data_model}_{timeOpt}_numeric_{num_patients}_{idx}.npy")
    ).view(-1, 1)
    data.num_relations = edge_type.max().item() + 1

    return data, patients, y


def train_model(model, data, lr, wd, max_epochs=2001, patience=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float("inf")
    min_delta = 1e-4
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ---- training ----
        model.train()
        optimizer.zero_grad()

        out = model(data)

        train_loss = torch.nn.functional.nll_loss(out[data.train_idx], data.train_y)
        # train_loss = criterion(out[data.train_idx], data.train_y)
        train_loss.backward()
        optimizer.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            out = model(data)

            val_loss = torch.nn.functional.nll_loss(out[data.valid_idx], data.valid_y)

            pred = out.argmax(dim=-1)

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
            TrainLoss=train_loss.item(),
            ValLoss=val_loss.item(),
            Train=train_acc,
            Val=val_acc,
            Test=test_acc,
        )

        if epochs_no_improve >= patience:
            log(EarlyStop=f"Stopped at epoch {epoch}")
            break

    # ---- load best model ----
    model.load_state_dict(torch.load(data.model_path, weights_only=True))
    return model


# ------------------------ Orchestrator ------------------------ #
def run_rgcn(
    num_patients,
    folds,
    time_opt,
    mcfg: ModelConfig,
    loader: LoaderConfig,
):
    result_dir = loader.results_dir

    # Load data
    data, patients, y = load_data(
        num_patients,
        mcfg.embed_dim,
        loader,
    )
    data.num_patients = num_patients
    data.model_path = (
        f"{result_dir}/model_weights_{time_opt}_{num_patients}.pth"
    )

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
        f"{result_dir}/metrics_{time_opt}_{num_patients}_hyperparams.csv",
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
    ) = k_fold(np.asarray(patients), y, folds)

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

        data.train_y = torch.Tensor(train_y).long().to(device)
        data.valid_y = torch.Tensor(val_y).long().to(device)
        data.test_y = torch.Tensor(test_y).long().to(device)

        # Initialize and train model
        model = RGCNNet(
            embed_dim=mcfg.embed_dim,
            hidden_dim=mcfg.hidden_dim,
            num_relations=data.num_relations,
            dropout=mcfg.dropout,
            num_classes=loader.num_classes,
            include_text_features=(loader.text_values_path is not None),
        ).to(device)
        model = train_model(model, data, lr=mcfg.lr, wd=mcfg.weight_decay)

        # Evaluate
        metric = evaluate_model(
            model, data, fold, result_dir, loader.data_mode, loader.classes, time_opt
        )
        all_metrics.append(metric)

    # Aggregate metrics
    panel = pd.concat(all_metrics)
    metrics_mean = panel.groupby(level=0).mean()
    metrics_mean.index.name = "MEAN"
    metrics_std = panel.groupby(level=0).std()
    metrics_std.index.name = "STD"

    mean_std_metrics(metrics_mean, metrics_std, loader.classes).to_csv(
        f"{result_dir}/metrics_{time_opt}_{num_patients}_mean_std.csv",
        sep="\t",
        index=False,
        mode="a",
    )
    metrics_mean.to_csv(
        f"{result_dir}/metrics_{time_opt}_{num_patients}.csv", mode="a"
    )
    metrics_std.to_csv(
        f"{result_dir}/metrics_{time_opt}_{num_patients}.csv", mode="a"
    )
