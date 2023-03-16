from ..core import Registry,build_from_cfg
DATASET=Registry("dataset")

def build_dataset(cfg,default_args=None):
    dataset=build_from_cfg(cfg,DATASET,default_args)
    return dataset

#def build_dataloader(dataset,):
