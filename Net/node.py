class Node():
    def __init__(self, innv):
        self.edgesIn = []  # list of edges in, should be sorted in increasing innovation order
        self.innv = innv
        self.val = 0.0
        self.visited = False

    def copyConstructor(self):
        return Node(self.innv)

    def __repr__(self):
        return f"({self.innv} {self.val})"
