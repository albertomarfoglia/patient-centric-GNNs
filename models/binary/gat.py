import torch
import torch.nn.functional as F
from torch.nn import Linear, PReLU, LayerNorm
from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        include_text_features: bool,
        num_relations = 0,
        num_heads: int = 4,
        text_dim: int = 384,
    ):
        super().__init__()

        self.include_text_features = include_text_features
        self.dropout = dropout
        self.num_heads = num_heads

        # ---- feature projections ----
        self.num_proj = Linear(1, embed_dim)

        if self.include_text_features:
            self.txt_proj = Linear(text_dim, embed_dim)

        self.node_proj = Linear(embed_dim, hidden_dim)

        self.input_act = PReLU(hidden_dim)

        # ---- GAT layers (NO edge types) ----
        self.conv1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )

        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )

        self.act1 = PReLU(hidden_dim)
        self.act2 = PReLU(hidden_dim)

        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)

        self.out = Linear(hidden_dim, 1)

    def forward(self, data):
        # ----- Numeric features -----
        num_mask = data.num_mask.view(-1, 1)
        h = self.num_proj(data.num_x * num_mask)

        # ----- Text features -----
        if self.include_text_features:
            txt_mask = data.txt_mask.view(-1, 1)
            txt = self.txt_proj(data.txt_x * txt_mask)
            h = h + txt

        h = h + self.node_proj(data.x)
        h = self.input_act(h)
        #h_res = h

        h = self.conv1(h, data.edge_index)
        h = self.norm1(h)
        h = self.act1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        #h = h + h_res

        h = self.conv2(h, data.edge_index)
        h = self.norm2(h)
        h = self.act2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        #h = h + h_res

        h = self.out(h)
        return h.squeeze(-1)