import networkx as nx

from dynamics import FlowDynamicBase


class MarkovChainDynamic(FlowDynamicBase):
    def __init__(self, graph: nx.Graph, departure_rate: float = 1.0):
        super().__init__(graph)

    @property
    def departure_rate(self):
        return 0.85

    def step(self, state, source=None): return state

    def apply(self, state, time, source=None): return state