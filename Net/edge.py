from dataclasses import dataclass


@dataclass
class Edge:
    nodeIn: 'Node'
    nodeOut: 'Node'
    innv: int
    weight: float = 1.0
    enable: bool = True

    def __repr__(self):
        return f"({self.nodeIn.innv},{self.nodeOut.innv},{self.weight},{self.enable},{self.innv})"

    def copyEdge(self, added):
        """Added: dict{nodeInnv->newNode}"""
        nodeIn = added[self.nodeIn.innv]
        nodeOut = added[self.nodeOut.innv]
        newEdge = Edge(nodeIn,nodeOut,self.innv,self.weight,self.enable)
        nodeIn.edgesOut.append(newEdge)
        nodeOut.edgesIn.append(newEdge)
        return newEdge

