import torch
from torch_geometric.nn import GCNConv,SAGEConv
import torch.nn.functional as F
from ..builder import ENCODER

@ENCODER.register_module()
class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        #self.conv3 = GCNConv(hidden_channels,num_classes)

    def forward(self, inputs,training=True,**kwargs):
        x = self.conv1(inputs.x,inputs.edge_index)
        x = x.relu()
        x = F.dropout(x, training=training)
        x = self.conv2(x, inputs.edge_index)
        return x