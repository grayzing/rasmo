import simulator

def main():
    sim = simulator.Simulator()
    sim.create_base_station()
    sim.create_user_equipment()
    sim.start()

main()