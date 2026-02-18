from simulation import Simulation, GnbSleepModeStrategy
import pandas as pd

def main():
    data: list[float] = []
    simulation: Simulation = Simulation(1, GnbSleepModeStrategy.Naive)
    simulation.initialize_network(19, 25)
    for step in range(1, 1001):
        data.append(simulation.total_energy_usage())
        simulation.step()
        simulation.time += simulation.delta
        

    df_power = pd.DataFrame(data, columns=["Energy Consumption (J)"])
    df_power.to_csv("./power_naive.csv", index=True)

main()