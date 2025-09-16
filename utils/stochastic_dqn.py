# Stochastic DQN Implementation by Gray

import torch
from torch_geometric import torch_geometric

class StochasticDQNModel(torch.nn.Module):
    def __init__(self, num_rus: int):
        super(StochasticDQNModel, self).__init__()
        self.num_rus = num_rus
        
        self.out_channels = 16
        self.in_channels = 1
        
        self.output_dim = 4
        
        self.gcn1 = torch_geometric.nn.GCNConv(self.in_channels, self.out_channels)
        self.gcn2 = torch_geometric.nn.GCNConv(self.out_channels, self.out_channels)
        self.linear = torch.nn.Linear(self.out_channels, self.output_dim) # Will output an N x 4 matrix.
        
    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        x = self.gcn1(x, edges)
        x = torch.relu(x)
        x = self.gcn2(x, edges)
        x = torch.relu(x)
        x = self.linear(x)
        return x