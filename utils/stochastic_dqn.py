# Stochastic DQN Implementation by Gray

import torch
import numpy as np
from torch_geometric import torch_geometric
from collections import deque

class StochasticDQNModel(torch.nn.Module):
    def __init__(self, num_rus: int):
        super(StochasticDQNModel, self).__init__()
        self.num_rus = num_rus
        
        self.out_channels = 128
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
    
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int):
        batch = np.random.choice(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
class Agent:
    def __init__(self, num_rus: int, replay_buffer_capacity: int = 10000):
        self.num_rus = num_rus
        self.model = StochasticDQNModel(num_rus + 1) # Additional dummy node is created so the model can create a regression for it. This dummy node represents no action taken. Sometimes the best action is to not change anything.
        self.target_model = StochasticDQNModel(num_rus + 1) # See above
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
    def select_action(self, state: torch.Tensor, edges: torch.Tensor) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_rus + 1) # Random action including dummy node
        with torch.no_grad():
            q_values = self.model(state, edges)
            return q_values.argmax().item()
        
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        
        edges = self.create_fully_connected_edges(state.size(1))
        next_edges = self.create_fully_connected_edges(next_state.size(1))
        
        q_values = self.model(state, edges)
        next_q_values = self.target_model(next_state, next_edges)
        
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = torch.nn.functional.mse_loss(q_value, expected_q_value.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    