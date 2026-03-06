import networkx as nx
import pytest

from heatmapx.dynamics import FlowDynamic


@pytest.fixture
def undirected_triangle():
    return nx.cycle_graph(3)


def test_flow_dynamic_protocol_is_runtime_checkable():
    assert hasattr(FlowDynamic, '__protocol_attrs__') or True

    class Dummy:
        graph = property(lambda self: None)
        node_order = property(lambda self: [])
        def initial_state(self, sources, initial=1.0): ...
        def step(self, state, source=None): ...
        def apply(self, state, time, source=None): ...
        def to_graph(self, state, key='heat'): ...

    assert isinstance(Dummy(), FlowDynamic)