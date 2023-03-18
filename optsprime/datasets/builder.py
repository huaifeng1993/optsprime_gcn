from ..core import Registry,build_from_cfg
from torch.utils.data import DataLoader
DATASET=Registry("dataset")

def build_dataset(cfg,default_args=None):
    dataset=build_from_cfg(cfg,DATASET,default_args)
    return dataset

def build_dataloader(dataset,data_cfg):
    data_loader=DataLoader(dataset,batch_size=data_cfg.batch_size,num_workers=data_cfg.num_workers,shuffle=True)
    return data_loader