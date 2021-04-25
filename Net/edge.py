from dataclasses import dataclass


@dataclass
class Edge:
    nodeIn: 'Node'
    nodeOut: 'Node'
    innv: int
    weight: float = 1.0
    enable: bool = True

    def __repr__(self):
        return f"({self.nodeIn.innv},{self.nodeOut.innv})"
