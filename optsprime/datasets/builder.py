from mmcv.utils import Registry,build_from_cfg
from torch_geometric.data import DataLoader
DATASET=Registry("dataset")

def build_dataset(cfg,default_args=None):
    dataset=build_from_cfg(cfg,DATASET,default_args)
    return dataset

def build_dataloader(dataset,cfg):
    data_loader=DataLoader(dataset,batch_size=cfg.batch_size,num_workers=cfg.num_workers,shuffle=True)
    return data_loader