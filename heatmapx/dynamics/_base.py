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

    def initial_state(self, sources, initial=1.0) -> npt.NDArray[np.floating]:
        return np.array([
            initial if node in sources else 0.0
            for node in self.node_order
        ])

    def to_graph(self, state: npt.NDArray) -> nx.MultiDiGraph:
        heated_graph = nx.MultiDiGraph(self._graph)
        nx.set_node_attributes(
            heated_graph,
            dict(zip(self.node_order, state)),
            name='heat'
        )
        return heated_graph