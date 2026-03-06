import networkx as nx
import numpy.typing as npt


class FlowDynamicBase:
    def __init__(self, graph: nx.Graph, **kwargs):
        self._graph = nx.MultiDiGraph(graph)

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    @property
    def node_order(self) -> list:
        return list(self._graph.nodes)