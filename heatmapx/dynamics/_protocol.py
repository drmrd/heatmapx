from typing import Protocol, runtime_checkable

import networkx as nx
import numpy.typing as npt


@runtime_checkable
class FlowDynamic(Protocol):
    """A simulator for network dynamics."""

    @property
    def graph(self) -> nx.MultiDiGraph: ...

    @property
    def node_order(self) -> list: ...

    def initial_state(self, sources, initial=1.0) -> npt.NDArray: ...

    def step(self, state, source=None) -> npt.NDArray: ...

    def apply(self, state, time, source=None) -> npt.NDArray: ...

    def to_graph(self, key='heat') -> nx.MultiDiGraph: ...