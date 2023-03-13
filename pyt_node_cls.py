from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pyg_utils import visualize_tsne
#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels,num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
def train(data,optimizer,model,criterion):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x,data.edge_index)  # Perform a single forward pass.
      # Compute the loss solely based on the training nodes.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()   # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test(data,model):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

def main():
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from optsprime.datasets.german import GermanDataset 
    #dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    dataset=GermanDataset(root='data/Genman',transform=NormalizeFeatures())
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    model = GCN(16,dataset.num_features,dataset.num_classes)
    print("可视化节点嵌入")
    model.eval()
    out = model(data.x, data.edge_index)
    visualize_tsne(out, color=data.y)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
    for epoch in range(1, 101):
        loss = train(data,optimizer,model,criterion)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test_acc = test(data,model)
    print(f'Test Accuracy: {test_acc:.4f}')
    print("可视化训练后节点嵌入")
    model.eval()
    out = model(data.x, data.edge_index)
    visualize_tsne(out, color=data.y)

if __name__=="__main__":
    main()