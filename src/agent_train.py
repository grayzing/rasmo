
from simulation import Simulation
from a2c_model import Actor, Critic, A2CAgent
from deepq_model import Agent
import torch

def main():

    agent: Agent = Agent(19, 190)

    agent.train(1000)

main()