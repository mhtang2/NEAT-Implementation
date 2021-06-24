from Net import Network, Edge, Node, Population, population
import numpy.random as random
import numpy as np
from Environments import stock_environment
import Net
import networkx as nx
import matplotlib.pyplot as plt

from Utils.timer import timer, resetTimer, printTimer, totalTime, getTimes
import time
# [i:0-4,b:5,hi:6,o:7,ho:8,hh:9-]


def trained_model_test():
    Network.setParams(5+1, 1, 20)
    pop = Population(100, 5+1, 1, 20, stock_environment.Stock_env)
    for epoch in range(500):
        start_time = time.perf_counter()

        stock_environment.Stock_env.setRandomStart()
        baseline = stock_environment.Stock_env.momentum_bot(
            stock_environment.Stock_env.trainingDat[stock_environment.Stock_env.random_stock], stock_environment.Stock_env.random_chunk_start, stock_environment.CHUNK)
        pop.setBaseline(baseline)
        pop.run()
        print(f"Momentum Fitness  {baseline}")
        print(f"Perfect Fitness  {stock_environment.Stock_env.perfect_bot()}")
        print(f"Chunk size {stock_environment.CHUNK}")
        print(f"Epoch {epoch}")
        print(f"Has {len(pop.population)} # of species")
        print(f"Has {pop.getCurrentPop()} # of members")
        print(f"Edge size {Network.edgeInnv.x}")
        print(f"Node size {Network.nodeInnv.x}")
        print(
            f"Average nodes per network: {np.mean([len([node for node in net.nodes if node.enabled])for species in pop.population for net in species.nets ])}")
        # printNetwork(pop.population[0].nets[0])
        total_time = time.perf_counter()-start_time
        getTimes()['Untimed'] = total_time-totalTime()
        printTimer(scale=total_time)
        print(f"Elapsed time: { total_time}")
        resetTimer()
    pop.validate()


def run():
    trained_model_test()
