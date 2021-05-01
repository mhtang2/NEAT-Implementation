from Net import Network, Edge, Node
import numpy.random as random

def resetNetwork(net):
    for node in net.nodes:
        node.val = 0
        node.newVal = 0
        node.visited = False

def printNetwork(net):
    print(net.nodes)
    print(net.edges)
    for node in net.nodes:
        p='['
        for edgeIn in node.edgesIn:
            if edgeIn.enable == True:
                p+=str(edgeIn.nodeIn.innv)+" "
        p+="] -> " +str(node.innv)
        print(p)

def multilayertest():
    n1 = Network(1, 1, 1)
    print("All nodes" + n1.nodes.__repr__())
    #[i,hi,o,ho]
    n1._add_edge(n1.nodes[0], n1.nodes[2], 3)
    n1._add_edge(n1.nodes[0], n1.nodes[3], 3)
    n1._add_edge(n1.nodes[1], n1.nodes[2], 3)
    n1._add_edge(n1.nodes[1], n1.nodes[3], 3)
    printNetwork(n1)

    print(n1.feedforward([1]))
    print(n1.feedforward([1]))
    print(n1.feedforward([1]))

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

# multilayertest()
for _ in range(100000):
    testRandom()
