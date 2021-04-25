class Node():
    def __init__(self, innv):
        self.edgesIn = []  # list of Edges
        self.innv = innv
        self.val = 0.0
        self.newVal = 0.0
        self.visited = False
    
    def __repr__(self):
        return str(self.innv)
