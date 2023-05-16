import torch
from torch.nn import Linear
import torch.nn.functional as F
from ..builder import ENCODER

@ENCODER.register_module()
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,proj_channels):
        super().__init__()
        self.lin1 = Linear(num_features, hidden_channels)
        #self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, inputs,training=True,**kwargs):
        x = self.lin1(inputs.x)
        x = x.relu()
        return x