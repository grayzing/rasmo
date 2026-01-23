from scheduler import Scheduler, Event, ActionType
from nr import NrGnb, NrUe

def main():
    scheduler: Scheduler = Scheduler(0.1)
    gnb1: NrGnb = NrGnb(0,0,0,25)
    gnb2: NrGnb = NrGnb(25, 0, 25, 25)

    ue1: NrUe = NrUe(12.5, 0, 12.5, 10, 1.5, scheduler, gnb1)

    scheduler.object_space.add_child(gnb1)
    scheduler.object_space.add_child(gnb2)
    scheduler.object_space.add_child(ue1)

    gnb1.attach_to_ue(ue1)

    scheduler.push_event(
            0.5, 
            ActionType.NrHandover, 
            {
                "ue": ue1,
                "source_gnb": gnb1,
                "destination_gnb": gnb2
            }
        )
    
    scheduler.run(5, verbose=True)

    




main()