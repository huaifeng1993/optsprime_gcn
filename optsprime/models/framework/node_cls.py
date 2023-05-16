
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

    def extract_feat(self, graph,**kwargs):
        out_puts={}
        x=self.encoder(graph,**kwargs)
        return x
    
    def forward(self, inputs, return_loss=True, **kwargs):
        if return_loss:
            outputs=self.forward_train(inputs,training=True,**kwargs)
        else:
            outputs=self.forward_test(inputs,training=False,**kwargs)
        return outputs
    
    def forward_test(self, inputs,**kwargs):
        outputs=self.extract_feat(inputs,**kwargs)
        if self.with_decoder:
            outputs=self.decoder(outputs,**kwargs)
        loss=self.loss(outputs,inputs.y,inputs.val_mask)
        outputs=F.sigmoid(outputs)
        return {"predict":outputs,"loss":loss["loss"]}
    
    def forward_train(self, inputs, **kwargs)->dict:
        outputs=self.extract_feat(inputs,**kwargs)
        if self.with_decoder:
            outputs=self.decoder(outputs,**kwargs)
        #TODO:计算损失函数
        loss=self.loss(outputs,inputs.y,inputs.train_mask)
        return loss

    def loss(self,pred,target,mask)->dict:
        #criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        criterion=torch.nn.BCEWithLogitsLoss(reduction="mean")
        cross_loss=criterion(pred[mask],target[mask][:,None].to(torch.float32))
        #dcie_loss=self.dice_loss(pred[mask],target[mask][:,None].to(torch.float32))
        #return {"loss":{"cross_loss":cross_loss,"dice_loss":dcie_loss}}
        return {"loss":{"cross_loss":cross_loss}}
    def dice_loss(self,input, target):
        smooth = 1.
        input=F.sigmoid(input)
        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / ((iflat * iflat).sum() + (tflat * tflat).sum() + smooth))
    
    # def metric_loss(self,y_pred,y_true):
    #     sig_pred = torch.nn.Softmax(dim=-1)(y_pred)
    #     cross_entroy_loss=torch.nn.CrossEntropyLoss()(y_pred,y_true)
    #     p_smaple=sig_pred[y_true==1][:,1].view(-1,1)
    #     n_sample=sig_pred[y_true==0][:,1].view(-1,1)    
    #     #
    #     pn_matrix=p_smaple-n_sample.transpose(1,0)
    #     pn_loss=torch.nn.functional.smooth_l1_loss(pn_matrix,torch.zeros_like(pn_matrix))
    #     #
    #     pp_matrix=torch.triu(p_smaple-p_smaple.transpose(1,0),diagonal=1)
    #     pp_loss=torch.nn.functional.smooth_l1_loss(pp_matrix,torch.zeros_like(pp_matrix))
    #     p1_loss=torch.nn.functional.smooth_l1_loss(p_smaple,torch.ones_like(p_smaple))
    #     return cross_entroy_loss+pn_loss