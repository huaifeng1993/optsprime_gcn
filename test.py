from optsprime import Config
from optsprime.datasets import build_dataset,build_dataloader
from optsprime.models import build_framework,build_encoder
from optsprime.optimizers import build_lrscheduler,build_optimizer

args=Config.fromfile("config/config_gcn.py")

print(args)
dataset=build_dataset(args.data.train)
data_loader=build_dataloader(dataset,args.data)
model=build_framework(args.model)
optimizer=build_optimizer(model,args.optimizer)
lr_schedule=build_lrscheduler(optimizer,args.lr_schedule)
print(len(dataset))