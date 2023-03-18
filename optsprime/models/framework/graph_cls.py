from .base import BaseFWork
from ..builder import FRAMEWORK,build_decoder,build_encoder
import warnings

@FRAMEWORK.register_module()
class GraphCls(BaseFWork):
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GraphCls).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
        #TODO:#加载预训练权重
        self.encoder=build_encoder(encoder)

        if decoder is not None:
            decoder_cfg=train_cfg.decoder if train_cfg is not None else None
            decoder_=decoder.copy()
            decoder_.update(train_cfg=decoder,test_cfg=test_cfg.decoder)
            self.encoder=build_decoder(decoder_)
        
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
    def extract_feat(self, graph):
        x=self.encoder(graph)
        return x 
    
    def forward(self, graph, data_metas, return_loss=True, **kwargs):

        return None
    
    def forward_test(self, graph, data_metas=None, **kwargs):
        
        return None
    
    def forward_train(self, graph, data_metas=None, **kwargs):
        x=self.encoder()
        return None