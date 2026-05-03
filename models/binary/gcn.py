import torch.nn.functional as F
from torch.nn import Linear, PReLU
from torch_geometric.nn import GCNConv
import torch

class GCNNet(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_relations: int,
        dropout: float,
        num_classes: int = 2,
        include_text_features: bool = False,
    ):
        super().__init__()

        self.include_text_features = include_text_features
        self.dropout = dropout

        self.num_proj = Linear(1, embed_dim)

        if self.include_text_features:
            self.text_projection = Linear(384, embed_dim)

        self.input_activation = PReLU(embed_dim)

        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 1)

        self.act1 = PReLU(hidden_dim)
        self.act2 = PReLU(hidden_dim)

    def forward(self, data):

        num_mask = data.num_mask.view(-1, 1)
        num_x = self.num_proj(data.num_x * num_mask)
        num_x = self.input_activation(num_x)

        h = num_x

        if self.include_text_features:
            txt_mask = data.txt_mask.view(-1, 1)
            txt = self.txt_proj(data.txt_x * txt_mask)
            txt = self.input_activation(txt)
            h = h + txt

        h = h + data.x

        h = self.conv1(h, data.edge_index)
        h = self.act1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, data.edge_index)
        h = self.act2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, data.edge_index)

        return h.squeeze(-1)
    