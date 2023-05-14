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
from ..utils import Evaluater
from .builder import DATASET
from sklearn.model_selection import train_test_split
import random
#torch.manual_seed(1236)
@DATASET.register_module()
class CVPADataset(InMemoryDataset):
	def __init__(self,
	             data_root,
				 split="random",
				 ratio_val: float=0.1,
				 ratio_test:float =0.1,
				 random_seed: int = 1234,
	             transform: Optional[Callable]=None,
	             pre_transform: Optional[Callable] = None):
		self.root = data_root
		super(CVPADataset, self).__init__(data_root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])
		self.split = split
		num_train=self.get(0).num_nodes
		num_val=int(ratio_val*num_train)
		num_test=int(ratio_test*num_train)
		random.seed(random_seed)
		assert self.split in ['random']
		if self.split == 'random':
			data = self.get(0)
			data.train_mask.fill_(False)
			data.val_mask.fill_(False)
			data.test_mask.fill_(False)
			idx=list(range(num_train))
			random.shuffle(idx)
			num_train=num_train-num_val-num_test
			train_idx=idx[:num_train]
			val_idx=idx[num_train:num_train+num_val]
			test_idx=idx[num_train+num_val:]
			data.train_mask[train_idx]=True
			data.val_mask[val_idx]=True
			data.test_mask[test_idx]=True
			self.data, self.slices = self.collate([data])
			
	@property
	def raw_file_names(self):
		return ['CVPA_preprocess_edge_index.csv','CVPA_preprocess_drop_branch_code.csv']
	@property
	def raw_dir(self):
		return os.path.join(self.root, 'raw')
	@property
	def processed_file_names(self):
		return ['cvpa_data_processed.pt']
	
	@property
	def processed_dir(self):
		return os.path.join(self.root, 'processed')

	def download(self):
		# raise NotImplementedError('Must indicate valid location of raw data. '
		#                           'No download allowed')
		pass
	def process(self):
		# raise NotImplementedError('Data is assumed to be processed')
		feature_df = pd.read_csv(os.path.join(self.raw_dir,"CVPA_preprocess_drop_branch_code.csv"))
		label=feature_df['overdue7']
		feature_x = feature_df.drop(['overdue7'], axis=1)
		feature_x=(feature_x-feature_x.min())/(feature_x.max()-feature_x.min())
		edge_df = pd.read_csv(os.path.join(self.raw_dir,'CVPA_preprocess_edge_index.csv'))
		
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
	
	def evaluate(self, input):
		"""
		return metric:dict
		"""
		input=input[0].view(-1).cpu().numpy()[self.data.val_mask.cpu().numpy()]
		target=self.data.y.cpu().numpy()[self.data.val_mask.cpu().numpy()]
		eval=Evaluater("cvpa",save_csv=False)
		metric=eval.clc_update(target,input)
		print(target.sum())
		eval.clean()
		return metric