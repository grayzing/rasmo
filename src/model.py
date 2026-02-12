
from torch_geometric import torch_geometric
from collections import deque
from simulation import Simulation

import torch

class Critic(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = 16
        self.conv1 = torch_geometric.nn.GCNConv(4, hidden_size)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_size,hidden_size)
        self.conv3 = torch_geometric.nn.GCNConv(hidden_size, 1)

    def forward(self, edges: torch.Tensor, weights: torch.Tensor, vertex_features: torch.Tensor) -> torch.Tensor:
        x = self.conv1(vertex_features, edges, weights)
        x = torch.relu(x)

        x = self.conv2(vertex_features, edges, weights)
        x = torch.relu(x)

        x = self.conv3(vertex_features, edges, weights)
        x = torch.relu(x)

        q_value = torch_geometric.nn.global_mean_pool.global_mean_pool(x, torch.tensor(len(vertex_features)))

        return q_value

class Actor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = 16
        self.conv1 = torch_geometric.nn.GCNConv(4, hidden_size)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_size,hidden_size)
        self.conv3 = torch_geometric.nn.GCNConv(hidden_size, 1)

    def forward(self, edges: torch.Tensor, weights: torch.Tensor, vertex_features: torch.Tensor) -> torch.Tensor:
        x = self.conv1(vertex_features, edges, weights)
        x = torch.relu(x)

        x = self.conv2(vertex_features, edges, weights)
        x = torch.relu(x)

        x = self.conv3(vertex_features, edges, weights)
        x = torch.relu(x)

        return x
    
class A2CAgent(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.actor = Actor()
        self.critic = Critic()

    def forward(self, vertex_embeddings: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor):
        policy = self.actor(vertex_embeddings, edges, weights)
        value = self.critic(vertex_embeddings, edges, weights)

    