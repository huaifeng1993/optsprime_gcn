from mmcv.utils import Registry,build_from_cfg
from torch_geometric.data import DataLoader
from .transform import build_pretransforms,build_transforms

DATASET=Registry("dataset")

def build_dataset(cfg,default_args=None):
    if hasattr(cfg,"transform"):
        transforms=build_transforms(cfg.transform)
        cfg.update(dict(transform=transforms))
    if hasattr(cfg,"pre_transform"):
        pre_transforms=build_pretransforms(cfg.pre_transform)
        cfg.update(dict(pre_transform=pre_transforms))
    dataset=build_from_cfg(cfg,DATASET,default_args)
    return dataset

def build_dataloader(dataset,cfg):
    data_loader=DataLoader(dataset,batch_size=cfg.batch_size,num_workers=cfg.num_workers,shuffle=True)
    return data_loader