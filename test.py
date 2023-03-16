from optsprime import Config
from optsprime.datasets import build_dataset
from optsprime.model import build_framework,build_encoder

args=Config.fromfile("config/config_gcn.py")
dataset=build_dataset(args.data.train)
model=build_framework(args.model)
print(len(dataset))