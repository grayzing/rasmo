
from torch_geometric import torch_geometric
from collections import deque
from simulation import Simulation
from nr import AdvancedSleepMode

from enum import Enum

import torch
import pandas as pd
import numpy as np

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
        return node_emb.squeeze()

class Actor(torch.nn.Module):
    def __init__(self, num_gnbs: int = 19, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = 64
        self.convlayers: torch.nn.ModuleList = torch.nn.ModuleList([
            torch_geometric.nn.GCNConv(6,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size,hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, hidden_size),
            torch_geometric.nn.GCNConv(hidden_size, hidden_size)
        ])

        self.lin = torch_geometric.nn.Linear(hidden_size,5)
        self.num_gnbs = num_gnbs

    def forward(self, vertex_features: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor, ) -> torch.Tensor:
        torch.autograd.set_detect_anomaly(True)
        for layer in self.convlayers:
            vertex_features = layer(x=vertex_features, edge_index=edges, edge_weight=weights)
            vertex_features = torch.sigmoid(vertex_features)

        node_emb: torch.Tensor = self.lin(vertex_features)[0:self.num_gnbs]
        return torch.log_softmax(node_emb.view(-1), 0) # So all elements of the matrix node_emb sum to one. Reshapes the tensor into a flat list

class A2CAgent:
    def __init__(self, num_gnbs: int) -> None:
        """
        Initialize Advantage Actor Critic agent.
        
        :param num_gnbs: Number of gNBs in the system.
        :type num_gnbs: int
        """
        self.actor = Actor() # Actor network, takes state and returns N x 5 matrix
        self.critic = Critic() # Critic network, takes state and returns a scalar

        self.actor_criterion: torch.nn.MSELoss = torch.nn.MSELoss()
        self.actor_optimizer: torch.optim.Adam = torch.optim.Adam(self.actor.parameters())

        self.critic_criterion: torch.nn.MSELoss = torch.nn.MSELoss()
        self.critic_optimizer: torch.optim.Adam = torch.optim.Adam(self.critic.parameters())

        self.num_gnbs = num_gnbs # N = num_gnbs by the way
        self.gamma = 0.99 # Discount factor
        self.entropy = 0.01

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
        return torch.tensor(reward + self.gamma * value).squeeze()

    def train(self, episodes: int = 1000, steps_per_episode: int = 100_000):
        """
        Train the A2C Agent
        
        :param episodes: Number of episodes to train A2C agent for
        :type episodes: int
        :param steps_per_episode: Number of actions A2C agent takes per episode
        :type steps_per_episode: int
        """
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        rewards: list[float] = []
        
        for e in range(episodes):
            simulation: Simulation = Simulation(1)
            simulation.initialize_network(19, 60)
            simulation.step()

            for step in range(steps_per_episode):
                if step % 500 == 0:
                    self.actor_optimizer.zero_grad()

                    x, edges, weights = simulation.get_state()

                    policy = self.actor(x, edges, weights) # Probability distribution of actions

                    action_distribution = torch.distributions.Categorical(policy)
                    action = action_distribution.sample([1,1]).squeeze() # Largest probability action from the num_gnbs x 5 vector returned by policy
                    action_row = int(action.item() // self.num_gnbs)
                    action_col = int(action.item() % 5)

                    log_prob = policy[action] # Don't have to do log since the policy is already log_softmax'ed.
                    
                    target_gnb = simulation.gnbs[action_row]
                    advanced_sleep_mode = AdvancedSleepModeIntMapping(action_col)
                    simulation.set_advanced_sleep_mode(target_gnb, advanced_sleep_mode) # Set ASM according to what log_prob tells us

                    time_to_wait = target_gnb.radio_unit.advanced_sleep_mode.value[0] + advanced_sleep_mode.value[0]

                    for _ in range(int(np.ceil(time_to_wait)) + 1): # Wait for ASM transition to finish.
                        # print("Waiting for ASM transition to complete")
                        simulation.step()

                    reward = simulation.reward() # Calculate reward from the ASM transition. Did QoS diminish, etc.
                    print("Reward: ", reward)
                    rewards.append(reward)

                    next_x, next_edges, next_weights = simulation.get_state()

                    self.critic_optimizer.zero_grad()
                    value = self.critic(next_x, next_edges, next_weights) # Approximate value from the action
                    print("Value approximation: ", value)
                    td = self.td(reward, value) # Approximate the advantage function using TD
                    print("TD: ", td)

                    critic_loss = self.critic_criterion(td, value) # Minimize the loss
                    critic_losses.append(critic_loss.item())
                    critic_loss.backward()

                    self.critic_optimizer.step()

                    actor_loss = -log_prob * td.detach() - self.entropy # Minimize the loss
                    actor_losses.append(actor_loss.item())

                    actor_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                    self.actor_optimizer.step()
                    
            simulation.screen.clearscreen()
        # Save the data from training to look at
        concat_data = {
            t : [rewards[t], actor_losses[t], critic_losses[t]] for t in range(len(rewards))
        }

        data = pd.DataFrame(concat_data, ["Step", "Reward", "Actor Loss", "Critic Loss"])
        data.to_csv("../data/results.csv")

        # Save the models to potentially reuse
        torch.save(self.critic.state_dict, "../models/critic.pt")
        torch.save(self.actor.state_dict, "../models/actor.pt")
                    
                    




        
    