from Net import Network, Edge, Node
n1 = Network(2, 2)
print("All nodes" + n1.nodes.__repr__())

n1._add_edge(n1.nodes[0], n1.nodes[2],3)
n1._add_edge(n1.nodes[0], n1.nodes[3])
n1._add_edge(n1.nodes[1], n1.nodes[2])
n1._add_edge(n1.nodes[1], n1.nodes[3])
n1._add_edge(n1.nodes[2],n1.nodes[3])

print(n1.feedforward([1,1]))
print(n1.feedforward([1,1]))
