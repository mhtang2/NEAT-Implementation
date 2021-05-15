from Net import Network, Edge, Node, Population
import numpy.random as random


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


def multilayertest():
    n1 = Network(1, 1, 1)
    n1._add_edge(n1.nodes[0], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[0], n1.nodes[3], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[3], 3)
    # n1.mutate_add_edge()
    n1.mutate_add_node()
    printNetwork(n1)
    n2 = Network(1, 1, 1)
    n2._add_edge(n1.nodes[0], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[0], n1.nodes[3], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[3], 3)
    # n1.mutate_add_edge()
    n2.mutate_add_node()
    printNetwork(n2)

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


#speciation_test()
