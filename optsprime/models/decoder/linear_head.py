import torch
from torch.nn import Sequential, Linear, ReLU
from ..builder import DECODER

@DECODER.register_module()
class LinearHead(torch.nn.Module):
    def __init__(self,input_proj_dim,proj_hidden_dim) -> None:
        super(LinearHead, self).__init__()
        self.proj_head = Sequential(Linear(input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
									   Linear(proj_hidden_dim, proj_hidden_dim))

        self.init_emb()
	
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self,x):
        x=self.proj_head(x)
        return x