class Node():
    def __init__(self, innv, enabled=True):
        self.edgesIn = []  # list of edges in, should be sorted in increasing innovation order
        self.innv = innv
        self.val = 0.0
        self.visited = False
        self.enabled = enabled

    def copyConstructor(self):
        return Node(self.innv, self.enabled)

    def __repr__(self):
        return f"({self.innv} {self.val})"
