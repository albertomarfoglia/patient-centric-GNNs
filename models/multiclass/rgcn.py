import torch.nn.functional as F
from torch.nn import Linear, PReLU
from torch_geometric.nn import RGCNConv
import torch

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

        self.num_proj = Linear(1, embed_dim)

        #self.node_proj = Linear(embed_dim, hidden_dim)

        self.input_activation = PReLU(embed_dim)

        self.conv1 = RGCNConv(embed_dim, hidden_dim, num_relations, num_bases=8)
        self.conv3 = RGCNConv(hidden_dim, num_classes, num_relations, num_bases=8)

        self.act1 = PReLU(hidden_dim)
        #self.act2 = PReLU(hidden_dim)

    def forward(self, data):
        num_mask = data.num_mask.view(-1, 1)
        num_x = self.num_proj(data.num_x * num_mask)
        num_x = self.input_activation(num_x)

        h = num_x

        if self.include_text_features:
            txt_mask = data.txt_mask.view(-1, 1)
            txt = self.input_activation(data.txt_x * txt_mask)
            h = h + txt

        h = h + data.x

        h = self.conv1(h, data.edge_index, data.edge_type)
        h = self.act1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, data.edge_index, data.edge_type)

        return F.log_softmax(h, dim=1)