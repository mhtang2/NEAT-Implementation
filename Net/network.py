from .counter import Counter
from .node import Node
from .edge import Edge
import numpy as np
import numpy.random as random


class Network():
    nodeInnv = Counter()
    edgeInnv = Counter()

    def __init__(self, numInputs, numOutputs, numRNN):
        # structure of nodes array: [i,hi,o,ho,hh]
        self.nodes = [Node(Network.nodeInnv.post())
                      for i in range(numInputs+numOutputs+2*numRNN)]
        self.edges = []
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.numRNN = numRNN

    # Helper to add Edge going between NodeIn and NodeOut, default weight to 1
    def _add_edge(self, nodeIn: 'Node', nodeOut: 'Node', weight=None, enable=True):
        if weight is None:
            weight = random.gauss(0, 1)
        newEdge = Edge(nodeIn, nodeOut,
                       Network.edgeInnv.post(), weight, enable)
        self.edges.append(newEdge)
        nodeIn.edgesOut.append(newEdge)
        nodeOut.edgesIn.append(newEdge)
        # Propogate dist changes
        nodeOut.updateDist(nodeIn.dist)
        return newEdge

    def _add_node(self):
        newNode = Node(Network.nodeInnv.post())
        self.nodes.append(newNode)
        return newNode

    # Pick two random nodes, add edge between them, edge cannot go to input node
    def mutate_add_edge(self):
        validConfig = False
        numNodes = len(self.nodes)
        numHidden = numNodes - (self.numInputs +
                                self.numOutputs+2*self.numRNN)
        while not validConfig:
            validConfig = True
            node1Num = (random.randint(numInputs+numRNN+numHidden) -
                        numHidden) % numNodes   # Pick from inputs or hidden
            # pick from outputs or hidden
            node2Num = random.randint(numInputs+numRNN, numNodes)
            if node1Num == node2Num:  # Same node
                validConfig = False
                continue
            # TODO: Check this tomorrow
            # If node2 is output node, it will be the ending node
            if not self.nodes[node2Num].edgesOut:
                nodeFrom = self.nodes[node1Num]
                nodeTo = self.nodes[node2Num]
            # Make a connection from
            elif (self.nodes[node1Num].dist < self.nodes[node2Num].dist):
                nodeFrom = self.nodes[node1Num]
                nodeTo = self.nodes[node2Num]
            else:
                nodeFrom = self.nodes[node2Num]
                nodeTo = self.nodes[node1Num]

            for edge in self.nodes[nodeToNum].edgesIn:
                if edge.NodeIn == self.nodes[nodeFromNum]:
                    validConfig = False
                    break
        self._add_edge(self.nodes[nodeFromNum], self.nodes[nodeToNum])

    ''' Pick edge, insert node between NodeIn and NodeOut of edge,
    weight between new node and NodeOut is 1, weight between NodeIn and
    new node is weight of old edge'''

    def mutate_add_node(self):
        #TODO: FIX THIS
        validConfig = False
        while not validConfig:
            edgeNum = random.randint(0, len(self.edges)-1)
            validConfig = self.edges[edgeNum].enable
        pickedEdge = self.edges[edgeNum]
        pickedEdge.enable = False
        newNode = self._add_node()
        self._add_edge(pickedEdge.nodeIn, newNode, weight=pickedEdge.weight)
        self._add_edge(newNode, pickedEdge.nodeOut, weight=1)

    # Return value inside node to pass through edge
    def _evalNode(self, node):
        if(node.visited):
            return node.val
        node.visited = True

        for edge in node.edgesIn:
            if edge.enable:
                node.val += edge.weight * self._evalNode(edge.nodeIn)

        return node.val

    # Two state feedforward, return prediction based on inputs
    def feedforward(self, inputValues):
        assert(len(inputValues) == self.numInputs)
        for i in range(self.numInputs):  # Set input nodes to values of inputs
            self.nodes[i].val = inputValues[i]
            self.nodes[i].visited = True

        output = np.empty(self.numOutputs)
        for i in range(self.numOutputs):
            outputNode = self.nodes[self.numInputs+self.numRNN+i]
            print("Evaluating output "+str(outputNode))
            output[i] = self._evalNode(outputNode)

        # Hidden outputs
        for i in range(self.numRNN):
            outputNode = self.nodes[self.numInputs +
                                    self.numRNN+self.numOutputs+i]
            print("Evaluating hidden output "+str(outputNode))
            self._evalNode(outputNode)
        # Hidden inputs
        for i in range(self.numRNN):
            self.nodes[self.numInputs+i].visited = True
            self.nodes[self.numInputs+i].val = self.nodes[self.numInputs +
                                                          self.numRNN + self.numOutputs + i].val

        # Reset graph for next time step
        for nodeNum in range(self.numInputs + self.numRNN, len(self.nodes)):
            node = self.nodes[nodeNum]
            node.val = 0
            node.visited = False

        return output
