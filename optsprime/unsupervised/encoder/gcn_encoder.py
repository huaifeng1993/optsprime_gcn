import numpy as np
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import Sequential, Linear, ReLU,Embedding
from torch_geometric.nn import global_add_pool,Linear
from ..convs import GINEConv,SAGEConv


class GCNEncoder(torch.nn.Module):
	def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="layerwise", is_infograph=False):
		super(GCNEncoder, self).__init__()

		self.pooling_type = pooling_type
		self.emb_dim = emb_dim
		self.num_gc_layers = num_gc_layers
		self.drop_ratio = drop_ratio
		self.is_infograph = is_infograph

		self.out_node_dim = self.emb_dim
		if self.pooling_type == "standard":
			self.out_graph_dim = self.emb_dim
		elif self.pooling_type == "layerwise":
			self.out_graph_dim = self.emb_dim * self.num_gc_layers
		else:
			raise NotImplementedError

		self.convs = torch.nn.ModuleList()
		self.bns = torch.nn.ModuleList()
		self.atom_encoder = torch.nn.Sequential(Linear(48,emb_dim))
		self.bond_encoder =torch.nn.Sequential(torch.nn.Linear(11,emb_dim))

		for i in range(num_gc_layers):
			# nn = Sequential(Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), ReLU(), Linear(2*emb_dim, emb_dim))
			# conv = GINEConv(nn) 
			conv = SAGEConv(emb_dim,emb_dim)
			bn = torch.nn.BatchNorm1d(emb_dim)
			self.convs.append(conv)
			self.bns.append(bn)
			

		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
		x = self.atom_encoder(x)
		edge_attr = self.bond_encoder(edge_attr)
		# compute node embeddings using GNN
		xs = []
		for i in range(self.num_gc_layers):
			x = self.convs[i](x, edge_index, edge_attr, edge_weight)
			x = self.bns[i](x)
			if i == self.num_gc_layers - 1:
				# remove relu for the last layer
				x = F.dropout(x, self.drop_ratio, training=self.training)
			else:
				x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
			xs.append(x)

		# compute graph embedding using pooling
		if self.pooling_type == "standard":
			xpool = global_add_pool(x, batch)
			return xpool, x

		elif self.pooling_type == "layerwise":
			xpool = [global_add_pool(x, batch) for x in xs]
			xpool = torch.cat(xpool, 1)
			if self.is_infograph:
				return xpool, torch.cat(xs, 1)
			else:
				return xpool, x
		else:
			raise NotImplementedError

	def get_embeddings(self, loader, device, is_rand_label=False):
		ret = []
		y = []
		train_mask = []
		test_mask = []
		val_mask = []
		with torch.no_grad():
			for data in loader:
				if isinstance(data, list):
					data = data[0].to(device)
				data = data.to(device)
				batch, x, edge_index, edge_attr,tr_mask,v_mask,te_msk = data.batch, data.x, data.edge_index, data.edge_attr,data.train_mask,data.val_mask,data.test_mask
				edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

				if x is None:
					x = torch.ones((batch.shape[0], 1)).to(device)
				_, x = self.forward(batch, x, edge_index, edge_attr, edge_weight)

				ret.append(x.cpu().numpy())
				if is_rand_label:
					y.append(data.rand_label.cpu().numpy())
				else:
					y.append(data.y.cpu().numpy())
				train_mask.append(tr_mask.cpu().numpy())
				test_mask.append(te_msk.cpu().numpy())
				val_mask.append(v_mask.cpu().numpy())
		ret = np.concatenate(ret, 0)
		y = np.concatenate(y, 0)
		train_mask = np.concatenate(train_mask, 0)
		test_mask = np.concatenate(test_mask, 0)
		val_mask = np.concatenate(val_mask, 0)
		return ret, y,train_mask,test_mask,val_mask
