import hypothesis as hyp
import hypothesis.strategies as st
import hypothesis_networkx as hyp_nx
import networkx as nx
import numpy as np
import pytest

from heatmapx.dynamics import FlowDynamic, FlowDynamicBase, MarkovChainDynamic


@pytest.fixture
def undirected_triangle():
    return nx.cycle_graph(3)


@pytest.fixture
def directed_triangle():
    return nx.cycle_graph(3, create_using=nx.DiGraph)


def random_digraph(min_nodes=2, max_nodes=20, weakly_connected=False):
    weights = st.floats(allow_nan=False, allow_infinity=False)
    node_data = st.fixed_dictionaries({
        'name': st.text(), 'number': st.integers(), 'weight': weights
    })
    edge_data = st.fixed_dictionaries({'weight': weights})

    return hyp_nx.graph_builder(
        graph_type=nx.DiGraph,
        node_keys=st.integers(),
        node_data=node_data,
        edge_data=edge_data,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        connected=weakly_connected
    )


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


def test_to_graph_node_values_match_state_vector(directed_triangle):
    dynamic = make_concrete_base(directed_triangle)
    state = np.array([1.0, 2.0, 3.0])

    heated_graph = dynamic.to_graph(state)

    for index, node in enumerate(dynamic.node_order):
        assert heated_graph.nodes[node]['heat'] == pytest.approx(state[index])


def test_markov_chain_satisfies_protocol(undirected_triangle):
    dynamic = MarkovChainDynamic(undirected_triangle)
    assert isinstance(dynamic, FlowDynamic)


def test_markov_chain_dynamic_accepts_departure_rate_parameter(undirected_triangle):
    dynamic = MarkovChainDynamic(undirected_triangle, departure_rate=0.85)
    assert dynamic.departure_rate == 0.85


def test_markov_chain_default_to_departure_rate_of_1(undirected_triangle):
    dynamic = MarkovChainDynamic(undirected_triangle)
    assert dynamic.departure_rate == 1.0


@hyp.given(G=random_digraph())
def test_markov_transition_matrix_columns_sum_to_one(G):
    departure_rate = 867.5309
    dynamic = MarkovChainDynamic(G, departure_rate=departure_rate)
    P = dynamic.transition_matrix

    col_sums = np.asarray(P.sum(axis=0)).flatten()
    for i, node in enumerate(dynamic.node_order):
        if G.out_degree(node) > 0 or G.in_degree(node) > 0:
            assert col_sums[i] == pytest.approx(1.0, abs=1e-12), (
                f'Column for node {node} sums to {col_sums[i]}, expected '
                f'{departure_rate}.'
            )


def test_transition_matrix_sink_node_is_absorbing():
    G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    dynamic = MarkovChainDynamic(G)
    P = dynamic.transition_matrix.toarray()

    sink_index = dynamic.node_order.index(3)
    assert P[sink_index, sink_index] == pytest.approx(1.0)


def test_markov_step_conserves_total_mass_without_source(directed_triangle):
    dynamic = MarkovChainDynamic(directed_triangle)
    state = dynamic.initial_state([0])

    new_state = dynamic.step(state)

    assert new_state.sum() == pytest.approx(state.sum())


def test_markov_step_moves_mass_along_edges():
    G = nx.path_graph(5, create_using=nx.DiGraph)
    dynamic = MarkovChainDynamic(G)
    state = dynamic.initial_state([0])

    new_state = dynamic.step(state)

    idx_0 = dynamic.node_order.index(0)
    idx_1 = dynamic.node_order.index(1)

    assert new_state[idx_0] == pytest.approx(0.0)
    assert new_state[idx_1] == pytest.approx(1.0)


def test_markov_step_supports_source_injection(directed_triangle):
    dynamic = MarkovChainDynamic(directed_triangle)
    state = dynamic.initial_state([0])
    source = dynamic.initial_state([0])

    new_state = dynamic.step(state, source=source)

    assert new_state.sum() == pytest.approx(2.0)


def test_markov_apply_given_zero_time_returns_same_state(directed_triangle):
    dynamic = MarkovChainDynamic(directed_triangle)
    state = dynamic.initial_state(list(directed_triangle.nodes)[:1])

    result = dynamic.apply(state, time=0)

    np.testing.assert_array_equal(result, state)


def test_markov_apply_one_step_equals_step(directed_triangle):
    dynamic = MarkovChainDynamic(directed_triangle)
    state = dynamic.initial_state(list(directed_triangle.nodes)[:1])

    next_state_from_step = dynamic.step(state)
    next_state_from_apply = dynamic.apply(state, time=1)

    np.testing.assert_allclose(next_state_from_apply, next_state_from_step)