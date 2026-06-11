import math
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.logging import log
import os
from configs.loader import LoaderConfig
from utils.gcn_utils import (
    k_fold,
    mean_std_metrics,
    evaluate_model,
)
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from codecarbon import EmissionsTracker
import time

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

    # Pin static tensors for fast async CPU→GPU transfers
    data.edge_index = data.edge_index.pin_memory() # type: ignore
    data.edge_type = data.edge_type.pin_memory()
    data.x = data.x.detach().pin_memory()
    data.num_x = data.num_x.pin_memory()
    data.num_mask = data.num_mask.pin_memory()
    if hasattr(data, "txt_x"):
        data.txt_x = data.txt_x.pin_memory()
        data.txt_mask = data.txt_mask.pin_memory()

    # Share static tensors in memory so worker threads don't copy them
    data.edge_index.share_memory_()
    data.edge_type.share_memory_()

    return data, patients, y


def _compute_binary_classification(model, data, optimizer):
    """
    Single forward pass: reuse training output for val/test metrics
    instead of running a second forward pass.

    NOTE: this assumes dropout/batchnorm differences between train/eval
    are acceptable for the val loss estimate. If not, switch back to a
    second forward pass scoped only to val/test indices.
    """
    out = model(data)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_loss = criterion(out[data.train_idx], data.train_y.float())
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        # Re-run in eval mode to get proper dropout/BN behaviour for val
        out_eval = model(data)
        val_loss = criterion(out_eval[data.valid_idx], data.valid_y.float())
        probs = torch.sigmoid(out_eval)
        pred = (probs > 0.5).long()

    return val_loss, pred


def _compute_multi_classification(model, data, optimizer):
    out = model(data)
    train_loss = torch.nn.functional.nll_loss(out[data.train_idx], data.train_y)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_eval = model(data)
        val_loss = torch.nn.functional.nll_loss(out_eval[data.valid_idx], data.valid_y)
        pred = out_eval.argmax(dim=-1)

    return val_loss, pred


def train_model(model, data, lr, wd, max_epochs=2001, patience=40, binary=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float("inf")
    min_delta = 1e-4
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        if binary:
            val_loss, pred = _compute_binary_classification(model, data, optimizer)
        else:
            val_loss, pred = _compute_multi_classification(model, data, optimizer)

        train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
        val_acc = float((pred[data.valid_idx] == data.valid_y).float().mean())
        test_acc = float((pred[data.test_idx] == data.test_y).float().mean())

        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), data.model_path)
        else:
            epochs_no_improve += 1

        log(
            Epoch=epoch,
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

import gc

def run_single_fold(
    fold: int,
    fold_split,
    base_data,
    mcfg,
    loader,
    excfg,
    result_dir: str,
    gpu_id: int,
):
    """
    Runs one cross-validation fold on the given GPU.

    Threads share the same process, so base_data tensors held in shared
    memory are accessed directly — no copies for the static graph structure.
    Each fold only allocates its own model and label tensors on the GPU.
    """
    tracker = EmissionsTracker(
        project_name=f"fold_{fold}",
        output_dir=result_dir,
        measure_power_secs=1
    )

    tracker.start()
    start = time.perf_counter()


    train_idx, val_idx, test_idx, train_y, val_y, test_y = fold_split

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Clone and move to device; pinned memory enables async non-blocking transfer
    data = base_data.clone().to(device, non_blocking=True)

    data.train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    data.valid_idx = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    data.test_idx = torch.as_tensor(test_idx, dtype=torch.long, device=device)

    data.train_y = torch.as_tensor(train_y, dtype=torch.long, device=device)
    data.valid_y = torch.as_tensor(val_y, dtype=torch.long, device=device)
    data.test_y = torch.as_tensor(test_y, dtype=torch.long, device=device)

    fold_dir = f"{result_dir}/{fold}"
    os.makedirs(fold_dir, exist_ok=True)
    data.model_path = f"{fold_dir}/model_weights.pth"

    model = excfg.model_type(
        embed_dim=mcfg.embed_dim,
        hidden_dim=mcfg.hidden_dim,
        num_relations=data.num_relations,
        dropout=mcfg.dropout,
        num_classes=loader.num_classes,
        include_text_features=excfg.include_text,
    ).to(device)

    model = train_model(
        model,
        data,
        lr=mcfg.lr,
        wd=mcfg.weight_decay,
        binary=(loader.num_classes == 2),
    )

    metric = evaluate_model(
        model,
        data,
        fold,
        result_dir,
        loader.data_mode,
        loader.classes,
        excfg.time_option,
    )

    del model
    del data
    gc.collect()
    torch.cuda.empty_cache()

    runtime = time.perf_counter() - start
    emissions = tracker.stop()

    print(f"Runtime: {runtime}", f"Emissions: {emissions}")

    # metric["RUNTIME_SEC"] = runtime
    # metric["EMISSIONS_KGCO2"] = emissions

    return fold, metric


def _make_gpu_queue(n_gpus: int, n_folds: int) -> Queue:
    """
    Round-robin GPU queue. Workers acquire a slot before running
    and release it when done, so at most n_gpus folds run concurrently.
    """
    # q: Queue = Queue()
    # for i in range(n_folds):
    #     q.put(i % n_gpus)
    # return q

    q = Queue()
    for gpu_id in range(n_gpus):
        q.put(gpu_id)
    return q


def run_gnn(num_patients, mcfg, loader, excfg):
    result_dir = loader.results_dir

    # Load graph once on CPU with pinned + shared memory
    data, patients, y = load_data(
        num_patients,
        mcfg.embed_dim,
        loader,
        excfg.include_text,
    )
    data.num_patients = num_patients

    # Save hyperparameters once
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

    # Split once
    (
        train_idx_list,
        val_idx_list,
        test_idx_list,
        train_y_list,
        val_y_list,
        test_y_list,
    ) = k_fold(np.asarray(patients), y, excfg.folds)

    fold_splits = list(
        zip(
            train_idx_list,
            val_idx_list,
            test_idx_list,
            train_y_list,
            val_y_list,
            test_y_list,
        )
    )

    base_data = data.cpu()

    n_gpus = max(torch.cuda.device_count(), 1)  # fallback to CPU if no GPU
    # Cap workers at available GPUs so no GPU is double-booked at once;
    # the queue below handles scheduling when folds > n_gpus.
    n_workers = min(excfg.folds, n_gpus)

    # GPU queue: each worker pops a GPU id, runs its fold, then returns the id.
    # This ensures at most n_gpus folds run simultaneously regardless of fold count.
    gpu_queue = _make_gpu_queue(n_gpus, excfg.folds)

    def _fold_task(fold, split):
        gpu_id = gpu_queue.get()
        try:
            return run_single_fold(
                fold,
                split,
                base_data,
                mcfg,
                loader,
                excfg,
                result_dir,
                gpu_id,
            )
        finally:
            gpu_queue.put(gpu_id)  # always release, even on exception

    all_metrics = [None] * excfg.folds

    # Threads share the same process → no CUDA re-init, no data serialization.
    # Pinned shared-memory tensors are accessed zero-copy across threads.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_fold_task, fold, split): fold
            for fold, split in enumerate(fold_splits)
        }
        for fut in as_completed(futures):
            fold, metric = fut.result()
            all_metrics[fold] = metric

    # Aggregate results in parent
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