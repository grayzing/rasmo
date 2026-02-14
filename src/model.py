
from torch_geometric import torch_geometric
from collections import deque
from simulation import Simulation
from nr import AdvancedSleepMode

from enum import Enum

import torch
import pandas as pd

def AdvancedSleepModeIntMapping(x) -> AdvancedSleepMode:
    assert x >= 0 and x <= 4

    if x == 0:
        return AdvancedSleepMode.ACTIVE
    
    if x == 1:
        return AdvancedSleepMode.SM1
    
    if x == 2:
        return AdvancedSleepMode.SM2
    
    if x == 3:
        return AdvancedSleepMode.SM3
    
    if x == 4:
        return AdvancedSleepMode.SM4
    
    else:
        raise IndexError
    
    


class Critic(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = 64
        self.convlayers: torch.nn.ModuleList = torch.nn.ModuleList([
            torch_geometric.nn.GCNConv(6,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, 1)
        ])

        self.parameter_list = torch.nn.ParameterList(layer.parameters() for layer in self.convlayers)

    def forward(self, vertex_features: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor, ) -> torch.Tensor:
        """
        Calculate value function given the state. Used for TD A2C algorithm
        
        :param vertex_features: N + M x 6 vector with node position, ASM (if gNB), PRB utilization (if gNB), and average throughput (if gNB)
        :type vertex_features: torch.Tensor
        :param edges: Associations between UEs and gNBs.
        :type edges: torch.Tensor
        :param weights: Normalized RSRP between UEs and gNBs.
        :type weights: torch.Tensor
        :return: Value function
        :rtype: Tensor
        """
        for layer in self.convlayers:
            vertex_features = layer(x=vertex_features, edge_index=edges, edge_weight=weights)
            vertex_features = torch.sigmoid(vertex_features)

        node_emb = torch_geometric.nn.pool.global_mean_pool(x=vertex_features, batch=None)
        return node_emb

class Actor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = 64
        self.convlayers: torch.nn.ModuleList = torch.nn.ModuleList([
            torch_geometric.nn.GCNConv(6,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, hidden_size)
        ])

        self.lin = torch_geometric.nn.Linear(hidden_size,5)

        self.parameter_list = torch.nn.ParameterList(layer.parameters() for layer in self.convlayers)

    def forward(self, vertex_features: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor, ) -> torch.Tensor:
        torch.autograd.set_detect_anomaly(True)
        for layer in self.convlayers:
            vertex_features = layer(x=vertex_features, edge_index=edges, edge_weight=weights)
            vertex_features = torch.sigmoid(vertex_features)

        node_emb: torch.Tensor = self.lin(vertex_features)
        return torch.log_softmax(node_emb, 0)

class A2CAgent:
    def __init__(self, num_gnbs: int) -> None:
        """
        Initialize Advantage Actor Critic agent.
        
        :param num_gnbs: Number of gNBs in the system.
        :type num_gnbs: int
        """
        self.actor = Actor()
        self.critic = Critic()

        self.actor_criterion: torch.nn.MSELoss = torch.nn.MSELoss()
        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(self.actor.parameters())

        self.critic_criterion: torch.nn.MSELoss = torch.nn.MSELoss()
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(self.critic.parameters())

        self.num_gnbs = num_gnbs
        self.gamma = 0.99 # Discount factor

    def td(self, reward: float, value: float) -> torch.Tensor:
        """
        Calculate TD
        
        :param reward: Reward at time t
        :type reward: float
        :param value: Value function approximation of reward
        :type value: float
        :return: TD
        :rtype: Tensor
        """
        return torch.tensor(reward + self.gamma * value)

    def train(self, episodes: int = 1000, steps_per_episode: int = 100_000_00):
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        rewards: list[float] = []
        
        for e in range(episodes):
            simulation: Simulation = Simulation(1)
            simulation.initialize_network(19, 60)
            simulation.step()

            current_vertex_embeddings, current_edges, current_weights = simulation.get_state()

            for step in range(steps_per_episode):
                if step % 500 == 0:
                    self.actor_optimizer.zero_grad()
                    policy = self.actor(current_vertex_embeddings, current_edges, current_weights)[0:self.num_gnbs]
                    action = torch.argmax(policy)

                    action_row, action_col = divmod(action.item(), policy.shape[1])
                    action_row = int(action_row)
                    action_col = int(action_col)

                    log_prob = policy[action_row][action_col]
                    
                    simulation.set_advanced_sleep_mode(simulation.gnbs[action_row], AdvancedSleepModeIntMapping(action_col))

                    for _ in range(499):
                        simulation.step()

                    reward = simulation.reward()
                    rewards.append(reward)

                    self.critic_optimizer.zero_grad()
                    value = self.critic(current_vertex_embeddings, current_edges, current_weights)

                    td = self.td(reward, value)

                    critic_loss = self.critic_criterion(td, value)
                    critic_losses.append(critic_loss.item())
                    critic_loss.backward()

                    self.critic_optimizer.step()

                    actor_loss = -log_prob * td.detach()
                    entropy = -(policy * torch.log(action + 1e-10)).sum()
                    actor_loss = actor_loss - 0.01 * entropy 

                    actor_losses.append(actor_loss.item())

                    actor_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                    self.actor_optimizer.step()

                    simulation.step()
                    current_vertex_embeddings, current_edges, current_weights = simulation.get_state()

        concat_data = {
            t : [rewards[t], actor_losses[t], critic_losses[t]] for t in range(len(rewards))
        }

        data = pd.DataFrame(concat_data, ["Step", "Reward", "Actor Loss", "Critic Loss"])
        data.to_csv("../data/results.csv")

                    
                    




        
    