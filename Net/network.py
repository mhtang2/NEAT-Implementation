from .counter import Counter
from .node import Node
from .edge import Edge
import numpy as np


class Network():
    nodeInnv = Counter()
    edgeInnv = Counter()

    def __init__(self, numInputs, numOutputs):
        self.nodes = [Node(Network.nodeInnv.post())
                      for i in range(numInputs+numOutputs)]
        self.edges = []
        self.numInputs = numInputs
        self.numOutputs = numOutputs

    def _add_edge(self,nodeIn,nodeOut,weight=1.0,enable=True):
        newEdge = Edge(nodeIn,nodeOut,self.edgeInnv.post(),weight,enable)
        nodeOut.edgesIn.append(newEdge)

    def mutate_add_edge(self):
        pass

    def mutate_add_node(self):
        pass

    def _evalNode(self, node):
        if(node.visited):
            return node.val
        node.visited = True
        if not node.edgesIn: # If empty list of input edges, return set input value
            return node.val

        node.newVal = 0
        for edge in node.edgesIn:
            node.newVal += edge.weight * self._evalNode(edge.nodeIn)

        return node.val
    def feedforward(self, inputValues):
        for i in range(self.numInputs): # Set input nodes to values of inputs
            self.nodes[i].val = inputValues[i]
        
        output = np.empty(self.numOutputs)
        for i in range(self.numOutputs):
            print("Evaluating output "+str(self.nodes[self.numInputs+i]))
            self._evalNode(self.nodes[self.numInputs+i])
            output[i] = self.nodes[self.numInputs+i].newVal
            

        for node in self.nodes: # Reset graph for next time step
            node.val = node.newVal
            node.visited=False
        
        return output
            
