"""Load preprocessed arrays into a PyTorch Geometric ``Data`` object."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch_geometric.data import Data

from .config import RunConfig
from .preprocess import CODE_IRI_PREFIX


@dataclass
class Graph:
    data: Data
    patient_nodes: np.ndarray  # node id of subject i at position i
    y: np.ndarray              # label of subject i
    meta: dict


def _read_entities(path: Path) -> list[str]:
    iris = []
    with open(path) as fh:
        for line in fh:
            idx, iri = line.rstrip("\n").split("\t", 1)
            assert int(idx) == len(iris), "entities.tsv must be ordered by id"
            iris.append(iri)
    return iris


def load_graph(run: RunConfig, cache: Path) -> Graph:
    iris = _read_entities(cache / "entities.tsv")
    index = {iri: i for i, iri in enumerate(iris)}
    try:
        patients = np.array([index[f"{run.subject_iri_prefix}{i}"] for i in range(run.num_patients)])
    except KeyError as err:
        raise KeyError(f"subject {err} of {run.run_id} not found in the graph") from err
    y = np.asarray(joblib.load(run.labels_path)).astype(int)
    if len(y) != run.num_patients:
        raise ValueError(f"{run.labels_path}: {len(y)} labels for {run.num_patients} patients")

    triples = torch.from_numpy(np.load(cache / "triples.npy"))
    # Every triple is reversed (tail -> head), as in the first submission.
    edge_index = torch.stack([triples[:, 2], triples[:, 0]])
    edge_type = triples[:, 1]

    def channel(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.load(cache / f"{name}.npy")).float().view(-1, 1)
        mask = (~torch.isnan(x)).float()
        return torch.nan_to_num(x, nan=0.0), mask

    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=len(iris))
    data.num_relations = int(edge_type.max()) + 1
    data.value_x, data.value_mask = channel("value")
    data.time_x, data.time_mask = channel("time")
    if run.features.include_text:
        data.txt_x = torch.from_numpy(np.load(cache / "text.npy")).float()
        data.txt_mask = (data.txt_x.abs().sum(dim=1) != 0).float()
    data.is_code = torch.tensor([iri.startswith(CODE_IRI_PREFIX) for iri in iris])
    meta = json.loads((cache / "meta.json").read_text())
    return Graph(data=data, patient_nodes=patients, y=y, meta=meta)
