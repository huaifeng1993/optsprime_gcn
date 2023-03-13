import torch
import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import train_test_split_edges
import os
from typing import Callable, List, Optional
torch.manual_seed(1236)
class GermanDataset(InMemoryDataset):
	def __init__(self,
	             root,
				 split="random",
				 num_train_per_class: int = 300,
				 ratio_val: float=0.2,
				 ratio_test:float =0.2,
	             transform: Optional[Callable]=None,
	             pre_transform: Optional[Callable] = None):
		self.root = root
		super(GermanDataset, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])
		self.split = split
		num_train=self.get(0).num_nodes
		num_val=int(ratio_val*num_train)
		num_test=int(ratio_test*num_train)
		
		assert self.split in ['public', 'full', 'random']#随机数种子没有确定
		if split == 'full':
			data = self.get(0)
			data.train_mask.fill_(True)
			data.train_mask[data.val_mask | data.test_mask] = False
			self.data, self.slices = self.collate([data])
		elif split == 'random':
			data = self.get(0)
			data.train_mask.fill_(False)
			for c in range(self.num_classes):
				idx = (data.y == c).nonzero(as_tuple=False).view(-1)
				idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
				data.train_mask[idx] = True
			remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
			remaining = remaining[torch.randperm(remaining.size(0))]
			data.val_mask.fill_(False)
			data.val_mask[remaining[:num_val]] = True
			data.test_mask.fill_(False)
			data.test_mask[remaining[num_val:num_val + num_test]] = True
			self.data, self.slices = self.collate([data])
			
	@property
	def raw_file_names(self):
		return ['german_edges.csv','german_features.csv']
	@property
	def raw_dir(self):
		return os.path.join(self.root, 'raw')
	@property
	def processed_file_names(self):
		return ['german_data_processed.pt']
	
	@property
	def processed_dir(self):
		return os.path.join(self.root, 'processed')

	def download(self):
		# raise NotImplementedError('Must indicate valid location of raw data. '
		#                           'No download allowed')
		pass
	def process(self):
		# raise NotImplementedError('Data is assumed to be processed')
		feature_df = pd.read_csv(os.path.join(self.raw_dir,"german_features.csv"))
		label=feature_df['label']
		feature_x = feature_df.drop(['PurposeOfLoan','label'], axis=1)
		edge_df = pd.read_csv(os.path.join(self.raw_dir,'german_edges.csv'))
		
		edge_index = torch.tensor(edge_df[['src','dst']].values,dtype=torch.long).t().contiguous()
		x = torch.tensor(feature_x.values,dtype=torch.float)
		y = torch.tensor(label.values,dtype=torch.long)
		train_index = torch.arange(y.size(0), dtype=torch.long)
		val_index = train_index
		test_index= train_index
		train_mask=self.index_to_mask(train_index,size=y.size(0))
		val_mask =self.index_to_mask(val_index, size=y.size(0))
		test_mask =self.index_to_mask(test_index, size=y.size(0))

		data_list = []
		data = Data(x=x, edge_index=edge_index, y=y)
		data.train_mask=train_mask
		data.val_mask=val_mask
		data.test_mask=test_mask
		data_list.append(data)
		data,slices = self.collate(data_list)
		#print(data.dtype)
		torch.save((data,slices),self.processed_paths[0])

	def index_to_mask(self,index, size):
		mask = torch.zeros((size, ), dtype=torch.bool)
		mask[index] = 1
		return mask