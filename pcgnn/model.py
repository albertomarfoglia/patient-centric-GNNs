"""RGCN+Lit: relational graph convolution with literal-aware node inputs.

With ``num_layers=2``, ``embeddings='fixed_random'`` and
``literal_mode='legacy_shared'`` this is the model of the first submission
(``models/binary/rgcn.py``): conv1 (d -> d), PReLU, conv_out (d -> 1).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGCNConv

from .config import FeatureConfig, ModelConfig


class NodeInput(nn.Module):
    """Initial node representation: base embedding + encoded literals."""

    def __init__(self, num_nodes: int, is_code: torch.Tensor, cfg: ModelConfig,
                 feats: FeatureConfig, generator: torch.Generator):
        super().__init__()
        d = cfg.embed_dim
        self.mode = cfg.embeddings
        self.literal_mode = feats.literal_mode
        self.include_text = feats.include_text

        base = torch.empty(num_nodes, d)
        nn.init.xavier_uniform_(base, gain=math.sqrt(2.0), generator=generator)
        if self.mode == "fixed_random":
            self.register_buffer("base", base)
        elif self.mode == "learnable_node":
            self.base = nn.Parameter(base)
        elif self.mode == "learnable_code":
            self.register_buffer("base", base)
            self.register_buffer("code_index", torch.nonzero(is_code).view(-1))
            self.code_emb = nn.Parameter(base[is_code].clone())
        else:
            raise ValueError(self.mode)

        self.value_proj = nn.Linear(1, d)
        self.has_time = self.literal_mode in ("separate", "separate_scaled", "scaled_relative")
        if self.has_time:
            self.time_proj = nn.Linear(1, d)
        # One PReLU shared by all literal encoders, as in the first submission.
        self.act = nn.PReLU(d)

    def forward(self, data) -> torch.Tensor:
        h = self.act(self.value_proj(data.value_x * data.value_mask))
        if self.has_time:
            h = h + self.act(self.time_proj(data.time_x * data.time_mask)) * data.time_mask
        if self.include_text:
            h = h + self.act(data.txt_x * data.txt_mask.view(-1, 1))
        if self.mode == "learnable_code":
            base = self.base.index_copy(0, self.code_index, self.code_emb)
        else:
            base = self.base
        return h + base


class RGCNLit(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, num_outputs: int, is_code: torch.Tensor,
                 cfg: ModelConfig, feats: FeatureConfig, generator: torch.Generator):
        super().__init__()
        self.inputs = NodeInput(num_nodes, is_code, cfg, feats, generator)
        dims = [cfg.embed_dim] + [cfg.hidden_dim] * (cfg.num_layers - 1) + [num_outputs]
        self.convs = nn.ModuleList(
            RGCNConv(a, b, num_relations, num_bases=cfg.num_bases) for a, b in zip(dims[:-1], dims[1:])
        )
        self.acts = nn.ModuleList(nn.PReLU(b) for b in dims[1:-1])
        self.dropout = cfg.dropout

    def forward(self, data) -> torch.Tensor:
        h = self.inputs(data)
        for i, conv in enumerate(self.convs):
            h = conv(h, data.edge_index, data.edge_type)
            if i < len(self.acts):
                h = F.dropout(self.acts[i](h), p=self.dropout, training=self.training)
        return h.squeeze(-1) if h.size(-1) == 1 else h
