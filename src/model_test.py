from simulation import Simulation
from model import Actor, Critic

def main():
    simulation: Simulation = Simulation(1)
    simulation.initialize_network(19, 25)
    for _ in range(100):
        simulation.step()

    x, edge_index, weights = simulation.get_state()
    print(x.size())
    print(edge_index.size())
    print(weights.size())

    actor: Actor = Actor()
    actor_vertex_output = actor(x, edge_index, weights)
    print(actor_vertex_output)

    critic: Critic = Critic()
    critic_graph_output = critic(x, edge_index, weights)
    # print(critic_graph_output)

main()