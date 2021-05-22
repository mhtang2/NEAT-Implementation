from Environments.environment import MEMORY_env, XOR_Env
from Net import Network, Edge, Node, Population, population
import numpy.random as random
import numpy as np
from Environments import XOR_Env
import Net
import networkx as nx
import matplotlib.pyplot as plt
random.seed(0)


def resetNetwork(net):
    for node in net.nodes:
        node.val = 0
        node.newVal = 0
        node.visited = False


def printNetwork(net):
    print("Nodes:", net.nodes)
    print("Edges:", net.edges)
    for node in net.nodes:
        p = '['
        for edgeIn in node.edgesIn:
            if edgeIn.enable == True:
                p += str(edgeIn.nodeIn.innv)+" "
        p += "] -> " + str(node.innv)
        print(p)
# TODO: Cycle detection test


def drawNetwork(net):
    G = nx.DiGraph()
    V = [node.innv for node in net.nodes]
    E = [(edge.nodeIn.innv, edge.nodeOut.innv) for edge in net.edges]
    G.add_nodes_from(V)
    G.add_edges_from(E)
    nx.draw_networkx(G)
    plt.show()


def multilayertest():
    n1 = Network(1, 1, 1)
    n1._add_edge(n1.nodes[0], n1.nodes[2], 3)
    # n1._add_edge(n1.nodes[0], n1.nodes[3], 3)
    # n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    # n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    # n1._add_edge(n1.nodes[1], n1.nodes[3], 3)
    # n1.mutate_add_edge()
    # n1.mutate_add_node()
    printNetwork(n1)

    '''
    print(n1.feedforward([1]))
    print(n1.nodes)
    print(n1.feedforward([1]))
    print(n1.nodes)
    print(n1.feedforward([1]))
    print(n1.nodes)
    '''


def testRandom():
    numInputs = 5
    numRNN = 5
    numHidden = 10
    numOutputs = 1
    numNodes = numInputs + 2 * numRNN + numOutputs
    # Pick from inputs(RNN too) and hidden
    node1Num = (random.randint(numInputs+numRNN +
                numHidden)-numHidden) % numNodes
    node2Num = random.randint(numInputs+numRNN, numNodes)
    assert((0 <= node1Num < numInputs+numRNN)
           or (numNodes-numHidden <= node1Num < numNodes))
    assert((numInputs+numRNN <= node2Num < numNodes))


def crossoverTest():
    numInputs = random.randint(1, 6)
    numRnn = random.randint(1, 6)
    numOutputs = random.randint(1, 6)
    numNets = 100
    nets = [None] * numNets
    for netIdx in range(numNets):
        net = Network(numInputs, numOutputs, numRnn, empty=False)
        newNodes = random.randint(1, 6)
        newEdges = random.randint(1, 6)
        for hiddenIdx in range(newEdges):
            net.mutate_add_edge()
        for hiddenIdx in range(newNodes):
            net.mutate_add_node()
        nets[netIdx] = net
    for i in range(100):
        for j in range(i+1, 100):
            newNet = Network.crossover(nets[i], nets[j])
            # Test innovation numbers all transfered over
            oldInnovation1 = set([edge.innv for edge in nets[i].edges])
            oldInnovation2 = set([edge.innv for edge in nets[j].edges])
            newInnovation = set([edge.innv for edge in newNet.edges])
            assert(len(oldInnovation2) > 0)
            assert(len(oldInnovation1) > 0)
            assert(len(newInnovation) > 0)
            assert(oldInnovation1.union(oldInnovation2).issubset(newInnovation))
            # Test edge innovations are in increasing order
            for i in range(1, len(newNet.edges)):
                assert(newNet.edges[i].innv > newNet.edges[i-1].innv)
            for node in newNet.nodes:
                edgesIn = node.edgesIn
                for i in range(1, len(edgesIn)):
                    assert(edgesIn[i].innv > edgesIn[i-1].innv)


def speciation_test():
    pop = Population(2, 1, 1, 1)
    nets = pop.networks
    print(pop.compatibilityDistance(nets[0][0], nets[0][1]))
    nets[0][0].mutate_add_edge()
    nets[0][0].mutate_add_edge()
    nets[0][0].mutate_add_node()
    nets[0][0].mutate_add_node()
    print(pop.compatibilityDistance(nets[0][0], nets[0][1]))


def insert_sorted_test():
    net = Network(1, 1, 1)
    for _ in range(100):
        e = Edge(net.nodes[0], net.nodes[0], random.randint(1000))
        net.insert_sorted(net.edges, e)
    for i in range(1, 100):
        assert(net.edges[i].innv >= net.edges[i-1].innv)


def population_test():
    pop = Population(200, 2+1, 1, 0, XOR_Env)
    for epoch in range(2000):
        pop.run()
        print(f"Epoch {epoch}")
        print(f"Has {len(pop.population)} # of species")
        print(f"Has {pop.getCurrentPop()} # of members")
        print(f"Edge size {Network.edgeInnv.x}")
        print(f"Node size {Network.nodeInnv.x}")
        print(
            f"Average nodes per network: {np.mean([len(net.nodes)for species in pop.population for net in species.nets ])}")
        # printNetwork(pop.population[0].nets[0])
    print(pop.population)

    biases = set([])
    # Check edges don't have same nodes
    edges = []
    for species in pop.population:
        for net in species.nets:
            for edge in net.edges:
                if(edge.nodeIn.innv == 2):
                    biases.add(edge.nodeOut.innv)
                edges.append(edge)
    print(f"Found {len(biases)} biases")
    for e1 in edges:
        for e2 in edges:
            if (e1.nodeIn.innv == e2.nodeIn.innv and e1.nodeOut.innv == e2.nodeOut.innv):
                assert(e1.innv == e2.innv)


Network.setParams(2+1, 1, 0)
population_test()
