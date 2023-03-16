
from .base import BaseFWork
from ..builder import FRAMEWORK,build_decoder,build_encoder
import warnings
import torch

@FRAMEWORK.register_module()
class GrapNodeCls(BaseFWork):
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GrapNodeCls,self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
        #TODO:#加载预训练权重
        self.encoder=build_encoder(encoder)
        if decoder is not None:
            self.decoder=build_decoder(decoder)
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
    def extract_feat(self, graph):
        x=self.encoder(graph)
        return x 
    
    # def forward(self, graph, data_metas, return_loss=True, **kwargs):

    #     return None
    
    def forward_test(self, graph, data_metas=None, **kwargs):
        x=self.encoder(graph)
        if self.with_decoder:
            x=self.decoder(x)
        return x
    
    def forward_train(self, graph, data_metas=None, **kwargs):
        x=self.encoder(graph)
        if self.with_decoder:
            x=self.decoder(x)
        #TODO:计算损失函数
        loss=self.loss(x,data_metas)
        return loss

    def loss(self,pred,data_metas):
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        bce_loss=criterion(pred[node_mask],targer[node_mask])
        return dict(BCE=bce_loss)