import networkx as nx
import pytest

from heatmapx.dynamics import FlowDynamic, FlowDynamicBase


@pytest.fixture
def undirected_triangle():
    return nx.cycle_graph(3)


@pytest.fixture
def directed_triangle():
    return nx.cycle_graph(3, create_using=nx.DiGraph)


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


def test_base_preserves_directed_graph_edges(directed_triangle):
    dynamic = make_concrete_base(directed_triangle)

    assert set(dynamic.graph.edges()) == set(directed_triangle.edges())


def test_base_preserves_node_weights(directed_triangle):
    for node, weight in enumerate(directed_triangle.nodes()):
        directed_triangle.nodes[node]['weight'] = weight

    dynamic = make_concrete_base(directed_triangle)

    for node, data in directed_triangle.nodes(data=True):
        assert dynamic.graph.nodes[node]['weight'] == data['weight']


def test_base_preserves_edge_weights(directed_triangle):
    for weight, edge in enumerate(directed_triangle.edges()):
        directed_triangle.edges[edge]['weight'] = weight

    dynamic = make_concrete_base(directed_triangle)

    for u, v, data in directed_triangle.edges(data=True):
        assert dynamic.graph.edges[u, v, 0]['weight'] == data['weight']


def test_node_order_contains_all_nodes(undirected_triangle):
    dynamic = make_concrete_base(undirected_triangle)
    assert set(dynamic.node_order) == set(undirected_triangle.nodes())