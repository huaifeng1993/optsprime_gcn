# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import GroupNorm, LayerNorm

from mmcv.utils import  build_from_cfg
from .builder import  OPTIMIZERS

