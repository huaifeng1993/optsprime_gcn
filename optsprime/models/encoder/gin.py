import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GATConv,GINConv
import torch.nn.functional as F
from ..builder import ENCODER

@ENCODER.register_module()
class GIN(torch.nn.Module):
    def __init__(self, num_features,hidden_channels,class_num,eps=1e-9):
        super().__init__()
        self.conv1 = GINConv(nn=Linear(num_features,hidden_channels), eps=eps)
        self.conv2 = GINConv(nn=Linear(hidden_channels, class_num), eps=eps)
    def forward(self, inputs,training=True,**kwargs):
        x = self.conv1(inputs.x,inputs.edge_index)
        x = x.relu()
        x = F.dropout(x,training=training)
        x = self.conv2(x, inputs.edge_index)
        return x