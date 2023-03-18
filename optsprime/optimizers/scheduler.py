import copy
import inspect
from typing import Dict, List
import torch
from ..core import Registry, build_from_cfg
LR_SCHEDULERS = Registry('lrcheduler')

def register_torch_optimizers() -> List:
    torch_lr_schedulers = []
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith('__'):
            continue
        _lrscheduler = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_lrscheduler) and issubclass(_lrscheduler,
                                                  torch.optim.lr_scheduler._LRScheduler):
            LR_SCHEDULERS.register_module()(_lrscheduler)
            torch_lr_schedulers.append(module_name)
    return torch_lr_schedulers

register_torch_optimizers()

def build_lrscheduler(optimizer,cfg:Dict):
    lr_schedulre_cfg=copy.copy(cfg)
    lr_schedulre_cfg["optimizer"]=optimizer
    return LR_SCHEDULERS.build(lr_schedulre_cfg)
