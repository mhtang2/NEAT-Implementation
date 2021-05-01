class Node():
    def __init__(self, innv):
        self.edgesIn = []  # list of edges in
        self.edgesOut = []  # list of edges out
        self.innv = innv
        self.val = 0.0
        self.dist: int = 0
        self.visited = False

    def updateDist(self, distIn):
        self.dist = max(self.dist, 1+distIn)
        for edge in self.edgesOut:
            edge.nodeOut.updateDist(self.dist)


    def copyConstructor(self):
        return Node(self.innv)

    def __repr__(self):
        return f"({self.innv} {self.val})"
