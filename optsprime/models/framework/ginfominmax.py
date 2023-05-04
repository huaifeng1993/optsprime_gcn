import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
import warnings
from .base import BaseFWork
from ..builder import FRAMEWORK,build_decoder,build_encoder

@FRAMEWORK.register_module()
class GInfoMinMax(BaseFWork):
    def __init__(self, 
                 encoder=None,
                 decoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GInfoMinMax,self).__init__(init_cfg)

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
        #TODO:#加载预训练权重
        self.encoder=build_encoder(encoder)
        if decoder is not None:
            self.decoder=build_decoder(decoder)
        self.pool = global_mean_pool
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def extract_feat(self, graph,**kwargs):
        x,node_emb=self.encoder(graph.batch,
                                graph.x,
                                graph.edge_index,
                                graph.edge_attr,
                                graph.edge_weight,
                                **kwargs)
        outputs=dict(x=x,
                    node_emb=node_emb)
        return outputs
    
    def forward(self, inputs, return_loss=True, **kwargs):
        if return_loss:
            outputs=self.forward_train(inputs,**kwargs)
        else:
            outputs=self.forward_test(inputs,**kwargs)
        return outputs
    
    def forward_test(self, inputs,**kwargs):
        outputs=self.extract_feat(inputs,**kwargs)
        if self.with_decoder:
            outputs=self.decoder(outputs)
        return outputs
    
    def forward_train(self, inputs, **kwargs):
        outputs=self.extract_feat(inputs,**kwargs)
        if self.with_decoder:
            outputs=self.decoder(outputs)
        #TODO:计算损失函数
        loss=self.loss(outputs,inputs.y,inputs.train_mask)
        return {"loss":loss}
    
    @staticmethod
    def loss( x, x_aug, temperature=0.2, sym=False):
        # x and x_aug shape -> Batch x proj_hidden_dim
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1)/2.0
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()

        return loss


