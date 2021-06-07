from Net import Network, Edge, Node, Population, population
import numpy.random as random
import numpy as np
from Environments import XOR_Env, MEMORY_env, Stock_env
import Net
import networkx as nx
import matplotlib.pyplot as plt

# [i:0-4,b:5,hi:6,o:7,ho:8,hh:9-]


def trained_model_test():
    pop = Population(100, 1+5, 1, 1, Stock_env)
    for epoch in range(200):
        pop.run()
        print(f"Epoch {epoch}")
        print(f"Has {len(pop.population)} # of species")
        print(f"Has {pop.getCurrentPop()} # of members")
        print(f"Edge size {Network.edgeInnv.x}")
        print(f"Node size {Network.nodeInnv.x}")
        print(
            f"Average nodes per network: {np.mean([len([node for node in net.nodes if node.enabled])for species in pop.population for net in species.nets ])}")
        # printNetwork(pop.population[0].nets[0])
    pop.test()

def naive_test():
    Network.setParams(5+1, 1, 1)
    network = Network(5+1, 1, 1, empty=False)
    env = Stock_env
    network._add_edge(network.nodes[6], network.nodes[7], 1)
    network._add_edge(network.nodes[0], network.nodes[8], 1)
    fitness = env.eval_test(network)
    print("Naive Test:", fitness)