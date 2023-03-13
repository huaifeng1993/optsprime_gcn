import torch
from torch.nn import Linear
import torch.nn.functional as F
from ..builder import ENCODER

@ENCODER.register_module()
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x