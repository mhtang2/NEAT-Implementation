from Net import Network, Edge, Node, Population, population
import numpy.random as random
import numpy as np
from Environments import stock_environment
import Net
import networkx as nx
import matplotlib.pyplot as plt

# [i:0-4,b:5,hi:6,o:7,ho:8,hh:9-]


def trained_model_test():
    Network.setParams(5+1, 1, 20)
    pop = Population(100, 5+1, 1, 20, stock_environment.Stock_env)
    for epoch in range(2000):
        stock_environment.Stock_env.setRandomStart()
        pop.run()
        print(f"Momentum Fitness  {stock_environment.Stock_env.momentum_bot()}")
        print(f"Perfect Fitness  {stock_environment.Stock_env.perfect_bot()}")
        print(f"Epoch {epoch}")
        print(f"Has {len(pop.population)} # of species")
        print(f"Has {pop.getCurrentPop()} # of members")
        print(f"Edge size {Network.edgeInnv.x}")
        print(f"Node size {Network.nodeInnv.x}")
        print(
            f"Average nodes per network: {np.mean([len([node for node in net.nodes if node.enabled])for species in pop.population for net in species.nets ])}")
        # printNetwork(pop.population[0].nets[0])
    pop.test()


def run():
    trained_model_test()
