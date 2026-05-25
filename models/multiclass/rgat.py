import torch
import torch.nn.functional as F
from torch.nn import Linear, PReLU
from torch_geometric.nn import GATConv, RGATConv

class RGATNet(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        include_text_features: bool,
        num_relations,
        heads: int = 1,
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
        self.conv1 = RGATConv(
            in_channels=embed_dim,
            out_channels=hidden_dim,
            heads=heads,
            num_relations=num_relations
        )

        self.conv2 = RGATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            num_relations=num_relations,
        )

        self.conv3 = RGATConv(
            in_channels=hidden_dim * heads,
            out_channels=num_classes,
            num_relations=num_relations,
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
        h = self.conv1(h, data.edge_index, data.edge_type)
        h = self.act1(h)
        #h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, data.edge_index, data.edge_type)
        h = self.act2(h)
        #h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, data.edge_index, data.edge_type)

        return F.log_softmax(h, dim=-1)
    

# class RGAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,
#                  num_relations):
#         super().__init__()
#         self.num_lin = torch.nn.Linear(1, in_channels)
#         self.conv1 = RGATConv(in_channels, hidden_channels, num_relations, heads=num_heads)
#         self.conv2 = RGATConv(hidden_channels*num_heads, hidden_channels, num_relations, heads=num_heads)
#         self.conv3 = RGATConv(hidden_channels*num_heads, hidden_channels, num_relations, heads=num_heads)
#         self.lin = torch.nn.Linear(hidden_channels, out_channels)  
        

#     def forward(self, num_x, x, edge_index, edge_type):
#         x = F.relu(self.num_lin(num_x))
#         x = x + data.x
#         x = self.conv1(x, edge_index, edge_type)
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.conv2(x, edge_index, edge_type)
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.conv3(x, edge_index, edge_type)
#         x = self.lin(x)        
#         return F.log_softmax(x, dim=-1)