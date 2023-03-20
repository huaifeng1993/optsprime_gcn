# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from . import (get_root_logger,find_latest_checkpoint,mkdir_or_exist,
                    get_time_str,LogBuffer,load_checkpoint,save_checkpoint,Config)
from ..datasets import build_dataloader
from ..optimizers import build_optimizer,build_lrscheduler
from torch.optim import Optimizer
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)
import os.path as osp
from abc import ABCMeta, abstractmethod
import warnings
import logging
from collections import OrderedDict

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class Runner():
    def __init__(self,
                 model: torch.nn.Module,
                 batch_processor: Optional[Callable] = None,
                 optimizer: Union[Dict, torch.optim.Optimizer, None] = None,
                 work_dir: Optional[str] = None,
                 logger: Optional[logging.Logger] = None,
                 meta: Optional[Dict] = None,
                 max_iters: Optional[int] = None,
                 max_epochs: Optional[int] = None) -> None:
        
        if batch_processor is not None:
            if not callable(batch_processor):
                raise TypeError('batch_processor must be callable, '
                                f'but got {type(batch_processor)}')
            warnings.warn(
                'batch_processor is deprecated, please implement '
                'train_step() and val_step() in the model instead.',
                DeprecationWarning)
            # raise an error is `batch_processor` is not None and
            # `model.train_step()` exists.
            _model = model
            if hasattr(_model, 'train_step') or hasattr(_model, 'val_step'):
                raise RuntimeError(
                    'batch_processor and model.train_step()/model.val_step() '
                    'cannot be both available.')
        else:
            assert hasattr(model, 'train_step')

        # check the type of `optimizer`
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, '
                        f'but optimizer["{name}"] is a {type(optim)}')
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f'optimizer must be a torch.optim.Optimizer object '
                f'or dict or None, but got {type(optimizer)}')

        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be a logging.Logger object, '
                            f'but got {type(logger)}')

        # check the type of `meta`
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(
                f'meta must be a dict or None, but got {type(meta)}')

        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        # create work_dir
        if isinstance(work_dir, str):
            self.work_dir: Optional[str] = osp.abspath(work_dir)
            mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
        self.timestamp = get_time_str()
        self.mode: Optional[str] = None
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        # TODO: Redesign LogBuffer, it is not flexible and elegant enough
        self.log_buffer = LogBuffer()
    
    @property
    def model_name(self) -> str:
        """str: Name of the model, usually the module class name."""
        return self._model_name
    
    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._iter
    
    @property
    def inner_iter(self) -> int:
        """int: Iteration in an epoch."""
        return self._inner_iter
    
    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs
    
    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(self, data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]], **kwargs) -> Any:
        pass

    @abstractmethod
    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str,
                        save_optimizer: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        
        save_checkpoint(self.model,
                        osp.join(out_dir,filename_tmpl),
                        self.optimizer,meta)

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def current_momentum(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums
    
    def load_checkpoint(
        self,
        filename: str,
        map_location: Union[str, Callable] = 'cpu',
        strict: bool = False,
        revise_keys: List = [(r'^module.', '')],
    ) -> Union[Dict, OrderedDict]:
        return load_checkpoint(
            self.model,
            filename,
            map_location,
            strict,
            self.logger,
            revise_keys=revise_keys)

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if self.meta is None:
            self.meta = {}
    
        # Re-calculate the number of iterations when resuming
        # models with different number of GPUs
        if 'config' in checkpoint['meta']:
            config =Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
                    previous_gpu_ids) != self.world_size:
                self._iter = int(self._iter * len(previous_gpu_ids) /
                                 self.world_size)
                self.logger.info('the iteration number is changed due to '
                                 'change of GPU number')

        # resume meta information meta
        self.meta = checkpoint['meta']

        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)