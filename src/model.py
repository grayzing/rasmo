
from torch_geometric import torch_geometric
from collections import deque
from simulation import Simulation

import torch

class Critic(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = 16
        self.convlayers: list[torch_geometric.nn.GCNConv] = [
            torch_geometric.nn.GCNConv(-1,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, 1)
        ]

        self.lin = torch_geometric.nn.Linear(hidden_size,1)

    def forward(self, vertex_features: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor, ) -> torch.Tensor:
        for layer in self.convlayers:
            vertex_features = layer(x=vertex_features, edge_index=edges, edge_weight=weights)
            vertex_features = torch.relu(vertex_features)

        node_emb = torch_geometric.nn.pool.global_mean_pool(x=vertex_features, batch=None)
        return node_emb

class Actor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = 64
        self.convlayers: list[torch_geometric.nn.GCNConv] = [
            torch_geometric.nn.GCNConv(6,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, hidden_size)
        ]

        self.lin = torch_geometric.nn.Linear(hidden_size,1)

    def forward(self, vertex_features: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor, ) -> torch.Tensor:
        torch.autograd.set_detect_anomaly(True)
        for layer in self.convlayers:
            vertex_features = layer(x=vertex_features, edge_index=edges, edge_weight=weights)
            vertex_features = torch.sigmoid(vertex_features)

        node_emb: torch.Tensor = self.lin(vertex_features)
        return torch.softmax(node_emb, 0)

class A2CAgent(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.actor = Actor()
        self.critic = Critic()

    def forward(self, vertex_embeddings: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor):
        policy = self.actor(vertex_embeddings, edges, weights)
        value = self.critic(vertex_embeddings, edges, weights)

    