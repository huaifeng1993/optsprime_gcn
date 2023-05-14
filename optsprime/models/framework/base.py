import torch
import numpy as np
from abc import ABCMeta,abstractmethod
from mmcv.runner import BaseModule
from mmcv.cnn import constant_init
from collections import OrderedDict
import torch.distributed as dist

class BaseFWork(BaseModule,metaclass=ABCMeta):
    def __init__(self,train_cfg=None,test_cfg=None,init_cfg=None):
        super(BaseFWork,self).__init__(init_cfg)
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg

    def  init_weights(self):
        super(BaseFWork, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)
                
    @property
    def with_encoder(self):
        return hasattr(self,"encoder") and self.encoder is not None
    
    @property
    def with_decoder(self):
        return hasattr(self,"decoder") and self.decoder is not None
    
    @abstractmethod
    def extract_feat(self,graph):
        pass

    @abstractmethod
    def loss(self,**kwargs):
        """Compute losses of the head."""
        pass

    def forward_train(self,graph,data_metas=None,**kwargs):
        pass

    def forward_test(self,graph,data_metas=None,**kwargs):
        pass

    def forward(self,graph,data_metas,return_loss=True,**kwargs):
        if return_loss:
            return self.forward_train(graph,data_metas,**kwargs)
        else:
            return self.forward_test(graph,data_metas,**kwargs)
    
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
    def train_step(self, data, optimizer):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.
                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        outsputs = self(**data)
        losses = outsputs['loss']
        loss, log_vars = self._parse_losses(losses)
        loss = dict(
            loss=loss, log_vars=log_vars)
        return {"loss":loss}
    
    def val_step(self, data, optimizer=None):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        out = self(**data,return_loss=False)
        predict = out['predict']
        losses=out['loss']
        loss, log_vars = self._parse_losses(losses)
        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value
        outputs = dict(predict=predict,
            loss=loss, log_vars=log_vars_)
        return outputs