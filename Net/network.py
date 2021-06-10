from .counter import Counter
from .node import Node
from .edge import Edge
import numpy as np
import numpy.random as random
from scipy.special import expit

from Net import edge


MAX_CYCLES_ADD_EDGE = 100
MAX_CYCLES_ADD_NODE = 100


def sigmoid(x):
    #return 1/(1 + np.exp(-x))
    return expit(x)

def tanh(x):
    return np.tanh(x)


def relu(x):
    return x if x > 0 else 0


activation = tanh


class Network():
    nodeInnv = Counter()
    edgeInnv = Counter()

    edgeGenome = dict()  # e = (n1,n2)  ->  edgeInnv
    nodeGenome = dict()  # e = (n1,n2)  -> nodeInnv

    @staticmethod
    def setParams(numInputs, numOutputs, numRNN):
        Network.nodeInnv.val(numInputs+numOutputs+2*numRNN)

    def __init__(self, numInputs, numOutputs, numRNN, empty=False):
        # structure of nodes array: [i,hi,o,ho,hh]
        if not empty:
            self.nodes = [Node(i)
                          for i in range(numInputs + numOutputs+2*numRNN)]
        else:
            self.nodes = []
        self.edges = []
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.numRNN = numRNN

    def numSetNodes(self):
        """Number of non-hidden nodes (Is fixed)"""
        return self.numOutputs+self.numInputs+2*self.numRNN

    # Helper to add Edge going between NodeIn and NodeOut, default weight to 1
    def _add_edge(self, nodeIn: 'Node', nodeOut: 'Node', weight=None, enable=True):
        if weight is None:
            weight = random.normal()
        edgeKey = (nodeIn.innv, nodeOut.innv)

        # Check if edge already exists in genome
        if edgeKey in Network.edgeGenome:
            newEdge = Edge(nodeIn, nodeOut,
                           Network.edgeGenome[edgeKey], weight, enable)
            self.insert_sorted(self.edges, newEdge)
            self.insert_sorted(nodeOut.edgesIn, newEdge)
        else:
            newEdge = Edge(nodeIn, nodeOut,
                           Network.edgeInnv.post(), weight, enable)

            self.edges.append(newEdge)
            nodeOut.edgesIn.append(newEdge)
            Network.edgeGenome[edgeKey] = newEdge.innv
        return newEdge

    def insert_sorted(self, a, obj):
        """Insert edge/node in a sorted array by innv"""
        lo = 0
        hi = len(a)
        while lo < hi:
            mid = (lo+hi)//2
            if a[mid].innv < obj.innv:
                lo = mid+1
            else:
                hi = mid
        a.insert(lo, obj)

    def _add_node(self, innv):
        newNode = Node(innv)
        self.insert_sorted(self.nodes, newNode)
        return newNode

    # Pick two random nodes, add edge between them, edge cannot go to input node
    def mutate_add_edge(self):
        validConfig = False
        numNodes = len(self.nodes)
        numHidden = numNodes - (self.numInputs +
                                self.numOutputs+2*self.numRNN)
        tries = 0
        while not validConfig:
            # Give up if can't find
            tries += 1
            if(tries > min(len(self.nodes), MAX_CYCLES_ADD_EDGE)):
                return

            validConfig = True
            # i hi o ho hh
            node1Num = (random.randint(self.numInputs+self.numRNN+numHidden) -
                        numHidden) % numNodes   # Pick from inputs or hidden
            # pick from outputs or hidden
            node2Num = random.randint(self.numInputs+self.numRNN, numNodes)
            if node1Num == node2Num:  # Same node
                validConfig = False
                continue
            # If node2 is output node, it will be the ending node
            elif (node2Num >= self.numInputs + self.numRNN) and (node2Num < self.numInputs + self.numRNN * 2 + self.numOutputs):
                nodeFrom = self.nodes[node1Num]
                nodeTo = self.nodes[node2Num]
            # If node1 is an input node, it will be the starting node
            elif node1Num < self.numInputs+self.numRNN:
                nodeFrom = self.nodes[node1Num]
                nodeTo = self.nodes[node2Num]
            else:
                nodeFrom = self.nodes[node2Num]
                nodeTo = self.nodes[node1Num]

            for edge in nodeTo.edgesIn:  # Edge already exists
                if edge.nodeIn == nodeFrom:
                    validConfig = False
                    break
        self._add_edge(nodeFrom, nodeTo)

    ''' Pick edge, insert node between NodeIn and NodeOut of edge,
    weight between new node and NodeOut is 1, weight between NodeIn and
    new node is weight of old edge'''

    def mutate_add_node(self):
        if(len(self.edges) == 0):
            self.mutate_add_edge()
            return
        validConfig = False
        tries = 0
        while not validConfig:
            tries += 1
            if(tries > min(len(self.edges), MAX_CYCLES_ADD_NODE)):
                return
            # Give up if too many tries
            edgeNum = random.randint(0, len(self.edges))
            validConfig = self.edges[edgeNum].enable
        pickedEdge = self.edges[edgeNum]
        pickedEdge.enable = False
        # Handle existing node
        edgeKey = (pickedEdge.nodeIn.innv, pickedEdge.nodeOut.innv)
        if (edgeKey in Network.nodeGenome):
            innv = Network.nodeGenome[edgeKey]
        else:
            innv = Network.nodeInnv.post()
            Network.nodeGenome[edgeKey] = innv

        newNode = self._add_node(innv)
        self._add_edge(pickedEdge.nodeIn, newNode, weight=pickedEdge.weight)
        self._add_edge(newNode, pickedEdge.nodeOut, weight=1)

    def mutate_node_ableness(self):
        if len(self.nodes) > self.numSetNodes():
            nodeNum = random.randint(self.numSetNodes(), len(self.nodes))
            self.nodes[nodeNum].enabled = not self.nodes[nodeNum].enabled

    def mutate_edge_ableness(self):
        if(len(self.edges) > 0):
            edgeNum = random.randint(len(self.edges))
            self.edges[edgeNum].enable = not self.edges[edgeNum].enable

    # Return value inside node to pass through edge
    def _evalNode(self, node):
        if not node.enabled:
            return 0
        if node.visited:
            return node.val
        node.visited = True

        for edge in node.edgesIn:
            if edge.enable:
                node.val += edge.weight * self._evalNode(edge.nodeIn)

        node.val = activation(node.val)  # Activation
        return node.val

    # Run one prediction
    def feedforward(self, inputValues):
        inputValues.append(1.0)
        assert(len(inputValues) == self.numInputs)
        for i in range(self.numInputs):  # Set input nodes to values of inputs
            self.nodes[i].val = inputValues[i]
            self.nodes[i].visited = True

        output = np.empty(self.numOutputs)
        for i in range(self.numOutputs):
            outputNode = self.nodes[self.numInputs+self.numRNN+i]
            # print("Evaluating output "+str(outputNode))
            output[i] = self._evalNode(outputNode)

        # Hidden outputs
        for i in range(self.numRNN):
            outputNode = self.nodes[self.numInputs +
                                    self.numRNN+self.numOutputs+i]
            # print("Evaluating hidden output "+str(outputNode))
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
