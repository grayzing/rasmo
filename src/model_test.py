from simulation import Simulation
from model import Actor, Critic, A2CAgent
import torch

def main():
    simulation: Simulation = Simulation(1)
    simulation.initialize_network(19, 25)
    for _ in range(100):
        simulation.step()

    x, edge_index, weights = simulation.get_state()
    print("Feature vector size: ", x.size())
    print("Edge vector size: ", edge_index.size())
    print("Weights size: ", weights.size())

    agent: A2CAgent = A2CAgent(19)

    actor_vertex_output = agent.actor(x, edge_index, weights)[0:19]
    print(actor_vertex_output)

    critic_graph_output = agent.critic(x, edge_index, weights)
    print(critic_graph_output)

    prob_dist = torch.distributions.Categorical(actor_vertex_output)
    action = prob_dist.sample()

    print(action)

main()