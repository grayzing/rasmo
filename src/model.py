from torch import nn, optim
import torch
from torch_geometric import torch_geometric
from collections import deque
import random
import copy
import scheduler
import datetime
import pandas as pd

class GraphDQN(nn.Module):
    def __init__(self) -> None:
        super(GraphDQN, self).__init__()
        self.device: torch.accelerator = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # type: ignore
        hidden_size = 8
        self.loss: nn.MSELoss = nn.MSELoss() 
        self.fc1: torch_geometric.nn.DenseGCNConv = torch_geometric.nn.DenseGCNConv(2, hidden_size)
        self.fc2: torch_geometric.nn.DenseGCNConv = torch_geometric.nn.DenseGCNConv(hidden_size, hidden_size)
        self.fc3: torch_geometric.nn.DenseGCNConv = torch_geometric.nn.DenseGCNConv(hidden_size, hidden_size)
        self.fcq: torch_geometric.nn.DenseGCNConv = torch_geometric.nn.DenseGCNConv(hidden_size, 1)
        
        
    def forward(self, x: torch.Tensor, e: torch.Tensor):
        x = torch.relu(self.fc1(x, e))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fcq(x, e)
        return x
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        torch.set_printoptions(threshold=10000)
        
    def add(self,experience):
        self.buffer.append(experience)
        
    def __sizeof__(self) -> int:
        return len(self.buffer)
        
    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        transitions = random.sample(self.buffer, batch_size)
        state_b, action_b, reward_b, next_state_b = zip(*transitions)
    
        state_b = torch.stack(state_b)                           # (batch, state_dim)
        action_b = torch.stack(action_b).long().unsqueeze(1)     # (batch, 1)
        reward_b = torch.stack(reward_b).float().unsqueeze(1)    # (batch, 1)
        next_state_b = torch.stack(next_state_b)                 # (batch, state_dim)
        
        return state_b, action_b, reward_b, next_state_b
                    
    def can_sample(self, batch_size)->bool:
        return len(self.buffer) >= batch_size * 10