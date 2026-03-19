import networkx as nx
import numpy as np
import numpy.typing as npt


class FlowBase:
    def __init__(self, graph: nx.Graph, **kwargs):
        self._graph = nx.MultiDiGraph(graph)

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    @property
    def node_order(self) -> list:
        return list(self._graph.nodes)

    def initial_state(
        self, sources: list | dict, initial=1.0
    ) -> npt.NDArray[np.floating]:
        initial_states = np.zeros((len(self._graph),))
        try:
            for source, initial_state in sources.items():
                initial_states[self.node_order.index(source)] = initial_state
        except AttributeError:
            for source in sources:
                initial_states[self.node_order.index(source)] = initial

        return initial_states

    def to_graph(
        self, state: npt.NDArray, key: str = 'heat'
    ) -> nx.MultiDiGraph:
        heated_graph = nx.MultiDiGraph(self._graph)
        nx.set_node_attributes(
            heated_graph, dict(zip(self.node_order, state)), name=key
        )
        return heated_graph
