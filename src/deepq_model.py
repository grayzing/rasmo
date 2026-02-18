import torch
import numpy as np
import pandas as pd
from random import sample, uniform, randint
from copy import deepcopy
from torch_geometric import torch_geometric
from simulation import Simulation
from nr import NrGnb, NrUe, AdvancedSleepMode, AsmTransitionState
from collections import deque

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

class QNetwork(torch.nn.Module):
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
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, vertex_features: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        torch.autograd.set_detect_anomaly(True)
        for layer in self.convlayers:
            vertex_features = layer(x=vertex_features, edge_index=edges, edge_weight=weights)
            vertex_features = torch.sigmoid(vertex_features)

        node_emb: torch.Tensor = self.lin(vertex_features)[0:self.num_gnbs]
        node_emb[self.num_gnbs:] = -np.inf #Action clipping Nr UEs (don't have to put these in advanced sleep modes)
        return node_emb
    
class StateContainer:
    def __init__(self, x: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor) -> None:
        self.x = x
        self.edges = edges
        self.weights = weights

class ReplayBuffer:
    def __init__(self) -> None:
        self.buffer_size = 20_000
        self.buffer: deque[tuple[StateContainer, int, float, StateContainer]] = deque()
    
    def add_experience(self, experience: tuple[StateContainer, int, float, StateContainer]) -> None:
        self.buffer.appendleft(experience)

    def sample(self, batch_size: int) -> list:
        return sample(self.buffer, batch_size)

class Agent:
    def __init__(self, num_gnbs = 19, num_ues = 190) -> None:
        self.num_gnbs: int = num_gnbs
        self.num_ues: int = num_ues

        self.batch_size: int = 64
        self.epsilon: float = 1
        self.min_epsilon: float = 0.01
        self.epsilon_decay_factor: float = 0.99
        self.target_network_update: int = 500
        self.gamma = 0.99

        self.replay_buffer: ReplayBuffer = ReplayBuffer()
        self.q: QNetwork = QNetwork(num_gnbs)
        self.q_target: QNetwork = deepcopy(self.q)
        self.kpis: list[tuple[float,float]] = []

    def generate_random_action(self) -> tuple[int,int]:
        return(randint(0,self.num_gnbs-1), randint(0,4))

    def train(self, episodes) -> None:
        for e in range(episodes):
            losses: list[float] = []
            rewards: list[float] = []

            simulation: Simulation = Simulation(1)
            simulation.initialize_network(19, self.num_ues)
            simulation.step()

            for t in range(1000):
                simulation.step()
                if t % 50 == 0:
                    x, edges, weights = simulation.get_state()
                    
                    # Epsilon-Greedy Strategy
                    action = None
                    target_gnb = None
                    sleep_mode = None
                    epsilon_probability = uniform(0,1)
                    if epsilon_probability <= self.epsilon:
                        gnb_id, asm_id = self.generate_random_action()
                        target_gnb = simulation.gnbs[gnb_id]
                        sleep_mode = AdvancedSleepModeIntMapping(asm_id)

                        action = self.num_gnbs * gnb_id + asm_id
                    else:
                        q_values: torch.Tensor = self.q(x, edges, weights)

                        # Action clipping O-RUs that are already in transition to another ASM
                        for g in range(self.num_gnbs):
                            if simulation.gnbs[g].radio_unit.asm_transition_state != AsmTransitionState.NONE:
                                q_values[g] = torch.tensor([-torch.inf]*5)

                        q_values = q_values.view(-1)
                        action_index = int(torch.argmax(q_values).item())

                        gnb_id = action_index // self.num_gnbs
                        asm_id = action_index % 5

                        target_gnb = simulation.gnbs[gnb_id]
                        sleep_mode = AdvancedSleepModeIntMapping(asm_id)

                        action = action_index
                    
                    simulation.set_advanced_sleep_mode(target_gnb, sleep_mode)

                    while target_gnb.radio_unit.asm_transition_state != AsmTransitionState.NONE and target_gnb.radio_unit.advanced_sleep_mode != sleep_mode: # Wait for the gNB to enter the desired sleep mode before we make calculations
                        simulation.step()
                
                    next_x, next_edges, next_weights = simulation.get_state()
                    reward = simulation.reward()
                    print("Reward, ", reward)
                    rewards.append(reward)

                    state_memory: tuple = (
                        StateContainer(x, edges, weights), 
                        action, 
                        reward, 
                        StateContainer(next_x, next_edges, next_weights)
                        )
                    
                    self.replay_buffer.add_experience(state_memory)

                    # Replay Learning Portion
                    if len(self.replay_buffer.buffer) >= self.batch_size:
                        sample = self.replay_buffer.sample(self.batch_size)

                        for experience in sample:
                            e_state, e_action, e_reward, e_next_state = experience
                            td_target = e_reward + (t != 999) * (self.gamma * torch.max(self.q_target(e_next_state.x,e_next_state.edges,e_next_state.weights)).item())
                            former_q_estimate = self.q(e_state.x, e_state.edges, e_state.weights)[e_action].item()
                            q_loss = self.q.criterion(td_target, former_q_estimate)

                            losses.append(q_loss)
                            print("Loss: ", q_loss)
                            q_loss.backward()
                            self.q.optimizer.step()
        
                    # Every C steps update the target network to be the current network
                    if t % self.target_network_update == 0:
                        self.q_target.load_state_dict(self.q.state_dict())

                    self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_factor) #Epsilon decay

            simulation.screen.clearscreen()

            if len(losses) > 0:
                avg_loss = sum(losses) / len(losses)
                avg_reward = sum(rewards) / len(rewards)

                self.kpis.append((avg_loss,avg_reward))

                data = pd.DataFrame(self.kpis, ["Q Loss", "Reward"])
                data.to_csv("../data/results"+str(e)+".csv")

            if e % 25 == 0:
                torch.save(self.q, "../models/q.pth")


