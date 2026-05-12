import torch.nn.functional as F
from torch.nn import Linear, PReLU
from torch_geometric.nn import GCNConv
import torch

# ------------------------ RGCN Model ------------------------ #
class GCNNet(torch.nn.Module):
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

        self.num_proj = Linear(1, embed_dim)

        self.node_proj = Linear(embed_dim, hidden_dim)

        if self.include_text_features:
            self.text_projection = Linear(384, embed_dim)

        self.input_activation = PReLU(embed_dim)

        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)

        self.act1 = PReLU(hidden_dim)
        self.act2 = PReLU(hidden_dim)

    def forward(self, data):
        num_x = self.num_proj(data.num_x * data.num_mask)
        num_x = self.input_activation(num_x)

        h = num_x

        if self.include_text_features:
            txt = self.txt_proj(data.txt_x * data.txt_mask.view(-1, 1))
            txt = self.input_activation(txt)
            h = h + txt

        h = h + self.node_proj(data.x)

        h = self.conv1(h, data.edge_index)
        h = self.act1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, data.edge_index)
        h = self.act2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, data.edge_index)

        return F.log_softmax(h, dim=1)