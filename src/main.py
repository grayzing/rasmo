from simulation import Simulation

def main():
    simulation: Simulation = Simulation(1)
    simulation.initialize_network(19, 25)
    simulation.run(100000, True)

main()