import torch
from torch_geometric.nn import GCNConv,GATConv,GINConv
import torch.nn.functional as F
from ..builder import ENCODER

@ENCODER.register_module()
class GAT(torch.nn.Module):
    def __init__(self, num_features,hidden_channels,heads,class_num):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_channels,heads=heads,concat=False)
        self.conv2 = GATConv(hidden_channels, class_num,heads=heads,concat=False)
    def forward(self, inputs,training=True,**kwargs):
        x = self.conv1(inputs.x,inputs.edge_index)
        x = x.relu()
        x = F.dropout(x,training=training)
        x = self.conv2(x, inputs.edge_index)
        return x