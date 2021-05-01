from Net import Network, Edge, Node
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
        p='['
        for edgeIn in node.edgesIn:
            if edgeIn.enable == True:
                p+=str(edgeIn.nodeIn.innv)+" "
        p+="] -> " +str(node.innv)
        print(p)
#TODO: Cycle detection test

def multilayertest():
    n1 = Network(1, 1, 1)
    n1._add_edge(n1.nodes[0], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[0], n1.nodes[3], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[3], 3)
    #n1.mutate_add_edge()
    n1.mutate_add_node()
    printNetwork(n1)
    n2 = Network(1, 1, 1)
    n2._add_edge(n1.nodes[0], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[0], n1.nodes[3], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    #n1._add_edge(n1.nodes[1], n1.nodes[3], 3)
    #n1.mutate_add_edge()
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
    node1Num = (random.randint(numInputs+numRNN+numHidden)-numHidden) % numNodes   # Pick from inputs(RNN too) and hidden
    node2Num =  random.randint(numInputs+numRNN,numNodes)
    assert((0<=node1Num<numInputs+numRNN) or (numNodes-numHidden<=node1Num<numNodes))
    assert((numInputs+numRNN<=node2Num<numNodes))

multilayertest()

