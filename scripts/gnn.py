import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

# Define the GNN model using Graph Convolutional Network (GCN)
class GNNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index, edge_attr):
        # Apply first graph convolution layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply second graph convolution layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Predict ratings or recommendation score
        x = self.linear(x)
        return x