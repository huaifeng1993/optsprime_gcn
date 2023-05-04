from optsprime import Config
from optsprime.datasets import build_dataset,build_dataloader
from optsprime.models import build_framework,build_encoder
from optsprime.optimizers import build_lrscheduler,build_optimizer
from optsprime.datasets.graphproppred import PygGraphPropPredDataset,PyGPPDataset
args=Config.fromfile("config/__base__/datasets/adg.py")
args=Config.fromfile("config/__base__/datasets/german.py")
# print(args)
dataset=build_dataset(args.data.train)
data_loader=build_dataloader(dataset,args.data)
for data in data_loader:
    print(data)
# model=build_framework(args.model)
# optimizer=build_optimizer(model,args.optimizer)
# lr_schedule=build_lrscheduler(optimizer,args.lr_schedule)
# print(len(dataset))

#dataset=PygGraphPropPredDataset(name="ogbg-molesol", root='./data/original_datasets/')
#dataset=PyGPPDataset(name="ogbg-molesol", root='./data/original_datasets/')
print(len(dataset))