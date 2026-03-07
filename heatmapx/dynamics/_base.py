import networkx as nx
import numpy as np
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

    def initial_state(self, sources) -> npt.NDArray[np.floating]:
        return np.array([
            1.0 if node in sources else 0.0
            for node in self.node_order
        ])