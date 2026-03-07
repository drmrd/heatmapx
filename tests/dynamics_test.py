import networkx as nx
import numpy as np
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


def test_node_order_is_deterministic_on_repetition(undirected_triangle):
    dynamic = make_concrete_base(undirected_triangle)
    assert dynamic.node_order == dynamic.node_order


def test_node_order_is_deterministic_across_instances(undirected_triangle):
    instance1_node_order = make_concrete_base(undirected_triangle).node_order
    instance2_node_order = make_concrete_base(undirected_triangle).node_order
    assert instance1_node_order == instance2_node_order


def test_initial_state_is_zero_except_at_sources(undirected_triangle):
    dynamic = make_concrete_base(undirected_triangle)
    source_node = list(undirected_triangle.nodes)[0]

    state = dynamic.initial_state([source_node])
    source_index = dynamic.node_order.index(source_node)

    assert state[source_index] == 1.0
    assert (state != 0.0).sum() == 1


def test_initial_state_respects_custom_initial_value(undirected_triangle):
    source_node = list(undirected_triangle.nodes)[0]

    dynamic = make_concrete_base(undirected_triangle)
    state = dynamic.initial_state([source_node], initial=5.0)

    assert state[dynamic.node_order.index(source_node)] == 5.0
    assert (state != 0.0).sum() == 1


def test_initial_state_supports_multiple_sources(undirected_triangle):
    dynamic = make_concrete_base(undirected_triangle)
    source_nodes = list(undirected_triangle.nodes)[1:]

    state = dynamic.initial_state(source_nodes, initial=5.0)
    assert state.sum() == 5.0 * len(source_nodes)
    assert (state != 0.0).sum() == len(source_nodes)


def test_initial_state_returns_numpy_array(undirected_triangle):
    dynamic = make_concrete_base(undirected_triangle)
    source_node = list(undirected_triangle.nodes)[0]

    state = dynamic.initial_state([source_node], initial=867.5309)

    assert isinstance(state, np.ndarray)
    assert state.dtype == np.floating or np.issubdtype(
        state.dtype, np.floating
    )


def test_to_graph_returns_a_new_graph_with_heat_attribute(directed_triangle):
    dynamic = make_concrete_base(directed_triangle)
    state = dynamic.initial_state([0])

    heated_graph = dynamic.to_graph(state)

    assert isinstance(heated_graph, nx.MultiDiGraph)
    assert heated_graph is not directed_triangle
    assert heated_graph is not dynamic.graph
    for node, node_heat in heated_graph.nodes(data='heat'):
        assert node_heat is not None, (
            f'Missing "heat" attribute on node {node}.'
        )


def test_to_graph_supports_a_custom_heat_key(directed_triangle):
    dynamic = make_concrete_base(directed_triangle)
    state = dynamic.initial_state([0])

    heated_graph = dynamic.to_graph(state, key='temperature')
    for node, heat in heated_graph.nodes(data='temperature'):
        assert heat is not None, f'Node {node} has no custom heat attribute.'