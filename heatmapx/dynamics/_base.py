import networkx as nx


class FlowDynamicBase:
    def __init__(self, graph: nx.Graph, **kwargs):
        self._graph = nx.MultiDiGraph(graph)

    @property
    def graph(self):
        return self._graph