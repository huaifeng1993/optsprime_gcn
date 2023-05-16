import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from ..builder import ENCODER

@ENCODER.register_module()
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels,normalize=True,improved=True)
        #self.conv2 = GCNConv(hidden_channels,hidden_channels,normalize=True)
        #self.conv3 = GCNConv(hidden_channels,num_classes)

    def forward(self, inputs,training=True,**kwargs):
        x = self.conv1(inputs.x,inputs.edge_index)
        # x = x.relu()
        # x = self.conv2(x, inputs.edge_index)
        #x = F.dropout(x, p=0.5, training=training)
        #x = x.relu()
        #x = self.conv3(x, inputs.edge_index)
        x =x.relu()
        return x