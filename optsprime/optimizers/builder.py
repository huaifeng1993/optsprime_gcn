# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import Dict, List
import torch
from ..core import Registry, build_from_cfg

OPTIMIZERS = Registry('optimizer')
def register_torch_optimizers() -> List:
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers
register_torch_optimizers()

def build_optimizer(model, cfg: Dict):
    optimizer_cfg=copy.deepcopy(cfg)
    optimizer_cfg["params"]=model.parameters()
    return OPTIMIZERS.build(optimizer_cfg)
