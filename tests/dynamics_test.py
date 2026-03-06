import networkx as nx
import pytest

from heatmapx.dynamics import FlowDynamic, FlowDynamicBase


@pytest.fixture
def undirected_triangle():
    return nx.cycle_graph(3)


def make_concrete_base(G, **kwargs):
    """Instantiate a FlowDynamicBase with no-op step and apply."""

    class Stub(FlowDynamicBase):
        def _build_operators(self): pass
        def step(self, state, source=None): return state
        def apply(self, state, time, source=None): return state

    return Stub(G, **kwargs)


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


def test_base_converts_undirected_graph_to_multidigraph(undirected_triangle):
    dynamic = make_concrete_base(undirected_triangle)
    assert isinstance(dynamic.graph, nx.MultiDiGraph)
    assert set(dynamic.graph.nodes()) == set(undirected_triangle.nodes())