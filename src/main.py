from scheduler import Scheduler

def main():
    scheduler: Scheduler = Scheduler(0.125)
    scheduler.initialize_network(16, 25)
    scheduler.run(1000, True)

main()