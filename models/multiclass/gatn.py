import torch
import torch.nn.functional as F
from torch.nn import Linear, PReLU
from torch_geometric.nn import GATConv

class GATNet(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        include_text_features: bool,
        num_relations,
        heads: int = 4,
    ):
        super().__init__()

        self.dropout = dropout
        self.include_text_features = include_text_features

        # --- feature projections ---
        self.num_proj = Linear(1, embed_dim)

        if include_text_features:
            self.text_projection = Linear(384, embed_dim)

        self.node_proj = Linear(embed_dim, hidden_dim)

        self.act1 = PReLU(hidden_dim * heads)
        self.act2 = PReLU(hidden_dim * heads)

        # --- GAT layers ---
        self.conv1 = GATConv(
            in_channels=embed_dim,
            out_channels=hidden_dim,
            heads=heads,
        )

        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
        )

        self.conv3 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=num_classes,
            heads=1,
            concat=False
        )

    def forward(self, data):

        # --- numeric features ---
        num_x = self.num_proj(data.num_x * data.num_mask.view(-1, 1))
        h = num_x

        # --- text features ---
        if self.include_text_features:
            txt = self.text_projection(
                data.txt_x * data.txt_mask.view(-1, 1)
            )
            h = h + txt

        # --- node features ---
        h = h + self.node_proj(data.x)

        # --- GAT layers ---
        h = self.conv1(h, data.edge_index)
        h = self.act1(h)
        #h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, data.edge_index)
        h = self.act2(h)
        #h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, data.edge_index)

        return F.log_softmax(h, dim=-1)