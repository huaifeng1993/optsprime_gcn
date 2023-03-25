
from .base import BaseFWork
from ..builder import FRAMEWORK,build_decoder,build_encoder
import warnings
import torch
import torch.nn.functional as F
@FRAMEWORK.register_module()
class GrapNodeCls(BaseFWork):
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GrapNodeCls,self).__init__(train_cfg,test_cfg,init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
        #TODO:#加载预训练权重
        self.encoder=build_encoder(encoder)
        if decoder is not None:
            self.decoder=build_decoder(decoder)
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg

    def extract_feat(self, graph,**kwargs):
        out_puts={}
        x=self.encoder(graph,**kwargs)
        return x
    
    def forward(self, inputs, return_loss=True, **kwargs):
        if return_loss:
            outputs=self.forward_train(inputs,**kwargs)
        else:
            outputs=self.forward_test(inputs,**kwargs)
        return outputs
    
    def forward_test(self, inputs,**kwargs):
        outputs=self.extract_feat(inputs,**kwargs)
        outputs=F.softmax(outputs)
        return outputs
    
    def forward_train(self, inputs, **kwargs):
        outputs=self.extract_feat(inputs,**kwargs)
        if self.with_decoder:
            outputs=self.decoder(outputs)
        #TODO:计算损失函数
        loss=self.loss(outputs,inputs.y,inputs.train_mask)
        return {"loss":loss}

    def loss(self,pred,target,mask):
        #criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        cross_loss=F.cross_entropy(pred[mask],target[mask])
        return cross_loss