from simulation import Simulation
from model import Actor, Critic, A2CAgent
import torch

def main():

    agent: A2CAgent = A2CAgent(19)

    agent.train()

main()