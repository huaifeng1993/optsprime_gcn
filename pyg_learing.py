from pyg_utils import visualize_embedding,visualize_graph
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

def train(data,optimizer,model,criterion):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

def main():
    import time
    from IPython.display import Javascript  # Restrict height of output cell.
    from IPython import display
    display.display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 430})'''))
    dataset = KarateClub()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]  # Get the first graph object.
    print(data)
    print('==============================================================')
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    edge_index = data.edge_index
    print(edge_index.t())
    print("===============================================================")
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)
    model = GCN(dataset.num_features,dataset.num_classes)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.
    for epoch in range(401):
        loss, h = train(data,optimizer,model,criterion)
        if epoch % 10 == 0:
            #visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
            #time.sleep(0.3)
            print("log==")
    visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)


if __name__ =="__main__":
    main()