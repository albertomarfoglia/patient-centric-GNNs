import os
import math
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.data import Data
from torch_geometric.logging import log
from torch_geometric.nn import RGCNConv

from utils import get_device, k_fold, compute_metrics, save_confusion_matrix, mean_std_metrics


# ------------------------ RGCN Model ------------------------ #
class RGCNNet(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_relations, dropout):
        super().__init__()
        self.num_lin = Linear(1, embed_dim)
        self.act_lin = torch.nn.PReLU(embed_dim)

        self.conv1 = RGCNConv(embed_dim, hidden_dim, num_relations, num_bases=8)
        self.act1 = torch.nn.PReLU(hidden_dim)

        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=8)
        self.act2 = torch.nn.PReLU(hidden_dim)

        self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=8)
        self.act3 = torch.nn.PReLU(hidden_dim)

        self.conv4 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=8)
        self.act4 = torch.nn.PReLU(hidden_dim)

        self.conv5 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=8)
        self.act5 = torch.nn.PReLU(hidden_dim)

        self.conv6 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=8)
        self.act6 = torch.nn.PReLU(hidden_dim)

        self.conv7 = RGCNConv(hidden_dim, 3, num_relations, num_bases=8)
        self.dropout = dropout

    def forward(self, data):
        x = self.act_lin(self.num_lin(data.num_x))
        x = x + data.x
        x = self.act1(self.conv1(x, data.edge_index, data.edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act2(self.conv2(x, data.edge_index, data.edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.act3(self.conv3(x, data.edge_index, data.edge_type))
        x = self.act4(self.conv4(x, data.edge_index, data.edge_type))
        x = self.act5(self.conv5(x, data.edge_index, data.edge_type))

        x = self.act6(self.conv6(x, data.edge_index, data.edge_type))

        x = self.conv7(x, data.edge_index, data.edge_type)
        return F.log_softmax(x, dim=1)

ROOT_URI_MAP = {
    "meds": "https://teamheka.github.io/meds-data/subject/",
    "sphn_pc": "http://nvasc.org/synth_patient_"
}

# ------------------------ Data Loading ------------------------ #
def load_data(num_patients, timeOpt, embed_dim, data_model):
    entity_df = pd.read_csv(f'processed_data/{data_model}_{timeOpt}_entities_{num_patients}.tsv', sep='\t', header=None)
    entity_dict = entity_df.set_index(entity_df[1]).to_dict()[0]

    patients = [entity_dict[f'<{ROOT_URI_MAP[data_model]}{i}>'] for i in range(num_patients)]
    y = np.asarray(joblib.load(f'data/outcomes_{data_model}_{timeOpt}_{num_patients}.joblib'))

    triples = pd.read_csv(f'processed_data/{data_model}_{timeOpt}_triples_{num_patients}.tsv', sep='\t', header=None)

    triples_inv = triples[[2, 1, 0]]
    triples_inv.columns = [0, 1, 2]
    triples = triples_inv

    num_x = torch.Tensor(np.load(f'processed_data/{data_model}_{timeOpt}_numeric_{num_patients}.npy'))

    edge_index = torch.vstack((torch.Tensor(triples[0]).long(), torch.Tensor(triples[2]).long()))
    edge_type = torch.Tensor(triples[1]).long()
    num_nodes = len(entity_dict)

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        num_classes=3
    )

    embedding = Parameter(torch.empty(num_nodes, embed_dim))
    torch.nn.init.xavier_uniform_(embedding, gain=math.sqrt(2.0))
    data.x = embedding
    data.num_x = num_x.view(-1, 1)
    data.num_relations = edge_type.max().item() + 1

    return data, patients, y


# ------------------------ Training ------------------------ #
# def train_model(model, data, lr, wd, max_epochs=500):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
#     best_val_acc = 0

#     for epoch in range(1, max_epochs + 1):
#         model.train()
#         optimizer.zero_grad()
#         out = model(data)
#         loss = torch.nn.functional.nll_loss(out[data.train_idx], data.train_y)
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         pred = out.argmax(dim=-1)
#         train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
#         val_acc = float((pred[data.valid_idx] == data.valid_y).float().mean())
#         test_acc = float((pred[data.test_idx] == data.test_y).float().mean())

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), data.model_path)

#         log(Epoch=epoch, Loss=loss.item(), Train=train_acc, Val=val_acc, Test=test_acc)

#     model.load_state_dict(torch.load(data.model_path, weights_only=True))
#     return model


def train_model(model, data, lr, wd, max_epochs=1000, patience=20):
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

        log(Epoch=epoch, TrainLoss=train_loss.item(), ValLoss=val_loss.item(), Train=train_acc, Val=val_acc, Test=test_acc)

        if epochs_no_improve >= patience:
            log(EarlyStop=f"Stopped at epoch {epoch}")
            break

    # ---- load best model ----
    model.load_state_dict(torch.load(data.model_path, weights_only=True))
    return model

# ------------------------ Evaluation ------------------------ #
def evaluate_model(model, data, fold, result_dir, data_model):
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_pred = out.argmax(dim=-1)
        y_prob = F.softmax(out, dim=-1)

    metrics = compute_metrics(data.test_y.cpu(), y_pred[data.test_idx].cpu(), y_prob[data.test_idx].cpu())
    # Save metrics to CSV
    metric_df = pd.DataFrame({
        'PRECISION': np.hstack((metrics['precision_class'], metrics['precision_macro'], metrics['precision_weighted'])),
        'RECALL': np.hstack((metrics['recall_class'], metrics['recall_macro'], metrics['recall_weighted'])),
        'F1SCORE': np.hstack((metrics['fscore_class'], metrics['fscore_macro'], metrics['fscore_weighted'])),
        'ACCURACY': np.hstack((np.zeros(4), metrics['accuracy'])),
        'AUC': np.hstack((metrics['auc_class'], metrics['auc_macro'], metrics['auc_weighted'])),
    }, index=['B2H', 'REHAB', 'DEATH', 'MACRO', 'WEIGHTED'])
    metric_df.index.name = f'Fold_{fold}'
    metric_df.to_csv(f"{result_dir}/metrics_{data_model}_{data.timeOpt}_{data.num_patients}.csv", mode='a')

    # Save confusion matrix
    save_confusion_matrix(
        data.test_y.cpu(),
        y_pred[data.test_idx].cpu(),
        f"{result_dir}/cm/cm_{data_model}_{data.timeOpt}_{data.num_patients}_{fold}.jpg",
        labels=["Back2Home", "Rehabilitation", "Death"]
    )

    return metric_df

# ------------------------ Orchestrator ------------------------ #
def run_rgcn(num_patients, folds, timeOpt, dr, lr, wd, embed_dim, hidden_dim, data_model="sphn_pc"):
    result_dir = f"results/{data_model}/exp_rgcn"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f'{result_dir}/cm', exist_ok=True)

    # Load data
    data, patients, y = load_data(num_patients, timeOpt, embed_dim, data_model)
    data.num_patients = num_patients
    data.timeOpt = timeOpt
    data.model_path = f'{result_dir}/model_weights_{data_model}_{timeOpt}_{num_patients}.pth'

    # Save hyperparameters
    hyper_param = pd.DataFrame({
        'DROPOUT': [dr],
        'LEARNING_RATE': [lr],
        'WEIGHT_DECAY': [wd],
        'EMBED_DIM': [embed_dim],
        'HIDDEN_DIM': [hidden_dim]
    })
    hyper_param.to_csv(f"{result_dir}/metrics_{data_model}_{timeOpt}_{num_patients}_hyperparams.csv", mode='a', index=False)

    # K-Fold split
    train_idx_list, val_idx_list, test_idx_list, train_y_list, val_y_list, test_y_list = k_fold(np.asarray(patients), y, folds)

    # Get device and move data
    device = get_device()
    data = data.to(device)

    all_metrics = []

    for fold, (train_idx, val_idx, test_idx, train_y, val_y, test_y) in enumerate(
        zip(train_idx_list, val_idx_list, test_idx_list, train_y_list, val_y_list, test_y_list)
    ):
        # Move all indices and labels to the correct device
        data.train_idx = torch.Tensor(train_idx).long().to(device)
        data.valid_idx = torch.Tensor(val_idx).long().to(device)
        data.test_idx = torch.Tensor(test_idx).long().to(device)

        data.train_y = torch.Tensor(train_y).long().to(device)
        data.valid_y = torch.Tensor(val_y).long().to(device)
        data.test_y = torch.Tensor(test_y).long().to(device)

        # Initialize and train model
        model = RGCNNet(embed_dim, hidden_dim, data.num_relations, dr).to(device)
        model = train_model(model, data, lr, wd)

        # Evaluate
        metric = evaluate_model(model, data, fold, result_dir, data_model)
        all_metrics.append(metric)

    # Aggregate metrics
    panel = pd.concat(all_metrics)
    metrics_mean = panel.groupby(level=0).mean()
    metrics_mean.index.name = 'MEAN'
    metrics_std = panel.groupby(level=0).std()
    metrics_std.index.name = 'STD'
				
    mean_std_metrics(metrics_mean, metrics_std).to_csv(f"{result_dir}/metrics_{data_model}_{timeOpt}_{num_patients}_mean_std.csv", sep="\t", index=False, mode='a')
    metrics_mean.to_csv(f"{result_dir}/metrics_{data_model}_{timeOpt}_{num_patients}.csv", mode='a')
    metrics_std.to_csv(f"{result_dir}/metrics_{data_model}_{timeOpt}_{num_patients}.csv", mode='a')
