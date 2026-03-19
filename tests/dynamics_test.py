import hypothesis as hyp
import hypothesis.strategies as st
import hypothesis_networkx as hyp_nx
import networkx as nx
import numpy as np
import pytest

from heatmapx.flows import (
    CapacityConstrainedFlow,
    Flow,
    FlowBase,
    MarkovChainFlow,
)


@pytest.fixture
def undirected_triangle():
    return nx.cycle_graph(3)


@pytest.fixture
def directed_triangle():
    return nx.cycle_graph(3, create_using=nx.DiGraph)


def random_digraph(
    min_nodes=2,
    max_nodes=15,
    node_data=None,
    edge_data=None,
    weakly_connected=False,
) -> st.SearchStrategy[nx.DiGraph]:
    default_weights = st.floats(
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        min_value=-1e154,
        max_value=1e154,
    )
    if node_data is None:
        node_data = st.fixed_dictionaries(
            {
                'name': st.text(),
                'number': st.integers(),
                'weight': default_weights,
            }
        )
    if edge_data is None:
        edge_data = st.fixed_dictionaries({'weight': default_weights})

    return hyp_nx.graph_builder(
        graph_type=nx.DiGraph,
        node_keys=st.integers(),
        node_data=node_data,
        edge_data=edge_data,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        connected=weakly_connected,
    )


def make_concrete_base(G, **kwargs):
    """Instantiate a FlowBase with no-op step and apply."""

    class Stub(FlowBase):
        def step(self, state, source=None):
            return state

        def apply(self, state, time, source=None):
            return state

    return Stub(G, **kwargs)


def test_flow_protocol_is_runtime_checkable():
    assert hasattr(Flow, '__protocol_attrs__') or True

    class Dummy:
        graph = property(lambda self: None)
        node_order = property(lambda self: [])

        def initial_state(self, sources, initial=1.0): ...
        def step(self, state, source=None): ...
        def apply(self, state, time, source=None): ...
        def to_graph(self, state, key='heat'): ...

    assert isinstance(Dummy(), Flow)


def test_base_converts_undirected_graph_to_multidigraph(undirected_triangle):
    flow = make_concrete_base(undirected_triangle)
    assert isinstance(flow.graph, nx.MultiDiGraph)
    assert set(flow.graph.nodes()) == set(undirected_triangle.nodes())


def test_base_preserves_directed_graph_edges(directed_triangle):
    flow = make_concrete_base(directed_triangle)

    assert set(flow.graph.edges()) == set(directed_triangle.edges())


def test_base_preserves_node_weights(directed_triangle):
    for node, weight in enumerate(directed_triangle.nodes()):
        directed_triangle.nodes[node]['weight'] = weight

    flow = make_concrete_base(directed_triangle)

    for node, data in directed_triangle.nodes(data=True):
        assert flow.graph.nodes[node]['weight'] == data['weight']


def test_base_preserves_edge_weights(directed_triangle):
    for weight, edge in enumerate(directed_triangle.edges()):
        directed_triangle.edges[edge]['weight'] = weight

    flow = make_concrete_base(directed_triangle)

    for u, v, data in directed_triangle.edges(data=True):
        assert flow.graph.edges[u, v, 0]['weight'] == data['weight']


def test_node_order_contains_all_nodes(undirected_triangle):
    flow = make_concrete_base(undirected_triangle)
    assert set(flow.node_order) == set(undirected_triangle.nodes())


def test_node_order_is_deterministic_on_repetition(undirected_triangle):
    flow = make_concrete_base(undirected_triangle)
    assert flow.node_order == flow.node_order


def test_node_order_is_deterministic_across_instances(undirected_triangle):
    instance1_node_order = make_concrete_base(undirected_triangle).node_order
    instance2_node_order = make_concrete_base(undirected_triangle).node_order
    assert instance1_node_order == instance2_node_order


def test_initial_state_is_zero_except_at_sources(undirected_triangle):
    flow = make_concrete_base(undirected_triangle)
    source_node = list(undirected_triangle.nodes)[0]

    state = flow.initial_state([source_node])
    source_index = flow.node_order.index(source_node)

    assert state[source_index] == 1.0
    assert (state != 0.0).sum() == 1


def test_initial_state_respects_custom_initial_value(undirected_triangle):
    source_node = list(undirected_triangle.nodes)[0]

    flow = make_concrete_base(undirected_triangle)
    state = flow.initial_state([source_node], initial=5.0)

    assert state[flow.node_order.index(source_node)] == 5.0
    assert (state != 0.0).sum() == 1


def test_initial_state_supports_multiple_sources(undirected_triangle):
    flow = make_concrete_base(undirected_triangle)
    source_nodes = list(undirected_triangle.nodes)[1:]

    state = flow.initial_state(source_nodes, initial=5.0)
    assert state.sum() == 5.0 * len(source_nodes)
    assert (state != 0.0).sum() == len(source_nodes)


def test_initial_state_supports_per_node_values(undirected_triangle):
    flow = make_concrete_base(undirected_triangle)

    state = flow.initial_state({0: 1.0, 2: 3.0})

    assert state[flow.node_order.index(0)] == pytest.approx(1.0)
    assert state[flow.node_order.index(1)] == pytest.approx(0.0)
    assert state[flow.node_order.index(2)] == pytest.approx(3.0)


def test_initial_state_sources_dict_overrides_initial_scalar_value(undirected_triangle):
    flow = make_concrete_base(undirected_triangle)

    state = flow.initial_state({0: 1.2}, initial=3.4)

    assert state[flow.node_order.index(0)] == pytest.approx(1.2)


def test_initial_state_returns_numpy_array(undirected_triangle):
    flow = make_concrete_base(undirected_triangle)
    source_node = list(undirected_triangle.nodes)[0]

    state = flow.initial_state([source_node], initial=867.5309)

    assert isinstance(state, np.ndarray)
    assert np.issubdtype(state.dtype, np.floating)


def test_to_graph_returns_a_new_graph_with_heat_attribute(directed_triangle):
    flow = make_concrete_base(directed_triangle)
    state = flow.initial_state([0])

    heated_graph = flow.to_graph(state)

    assert isinstance(heated_graph, nx.MultiDiGraph)
    assert heated_graph is not directed_triangle
    assert heated_graph is not flow.graph
    for node, node_heat in heated_graph.nodes(data='heat'):
        assert node_heat is not None, (
            f'Missing "heat" attribute on node {node}.'
        )


def test_to_graph_supports_a_custom_heat_key(directed_triangle):
    flow = make_concrete_base(directed_triangle)
    state = flow.initial_state([0])

    heated_graph = flow.to_graph(state, key='temperature')
    for node, heat in heated_graph.nodes(data='temperature'):
        assert heat is not None, f'Node {node} has no custom heat attribute.'


def test_to_graph_node_values_match_state_vector(directed_triangle):
    flow = make_concrete_base(directed_triangle)
    state = np.array([1.0, 2.0, 3.0])

    heated_graph = flow.to_graph(state)

    for index, node in enumerate(flow.node_order):
        assert heated_graph.nodes[node]['heat'] == pytest.approx(state[index])


def test_markov_chain_satisfies_protocol(undirected_triangle):
    flow = MarkovChainFlow(undirected_triangle)
    assert isinstance(flow, Flow)


@pytest.mark.parametrize('offending_weight', (np.inf, -np.inf, np.nan))
def test_markov_raises_on_nonfinite_edge_weights(offending_weight):
    with pytest.raises(ValueError, match='finite'):
        MarkovChainFlow(nx.DiGraph([(0, 1, {'weight': offending_weight})]))


@hyp.given(G=random_digraph())
def test_markov_transition_matrix_columns_sum_to_one(G):
    flow = MarkovChainFlow(G)
    P = flow.transition_matrix

    col_sums = np.asarray(P.sum(axis=0)).flatten()
    for i, node in enumerate(flow.node_order):
        if G.out_degree(node) > 0 or G.in_degree(node) > 0:
            assert col_sums[i] == pytest.approx(1.0, abs=1e-12), (
                f'Column for node {node} sums to {col_sums[i]}, expected 1.0.'
            )


def test_markov_raises_on_transition_matrix_weight_overflow():
    G = nx.DiGraph([(0, 1, {'weight': 1e308}), (0, 2, {'weight': 1e308})])
    flow = MarkovChainFlow(G, weight='weight')

    with pytest.raises(ValueError, match='overflow'):
        flow.transition_matrix


def test_markov_raises_on_transition_matrix_weight_underflow():
    G = nx.DiGraph([(0, 1, {'weight': 5e-324})])
    flow = MarkovChainFlow(G, weight='weight')

    with pytest.raises(ValueError, match='too small'):
        flow.transition_matrix


def test_markov_transition_matrix_sink_node_is_absorbing():
    G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    flow = MarkovChainFlow(G)
    P = flow.transition_matrix.toarray()

    sink_index = flow.node_order.index(3)
    assert P[sink_index, sink_index] == pytest.approx(1.0)


def test_markov_transition_matrix_is_cached(directed_triangle, mocker):
    flow = MarkovChainFlow(directed_triangle)

    to_scipy_sparse_array_mock = mocker.patch(
        'heatmapx.flows.markov.nx.to_scipy_sparse_array',
        wraps=nx.to_scipy_sparse_array,
    )

    flow.transition_matrix
    flow.transition_matrix

    to_scipy_sparse_array_mock.assert_called_once()


def test_markov_step_conserves_total_mass_without_source(directed_triangle):
    flow = MarkovChainFlow(directed_triangle)
    state = flow.initial_state([0])

    new_state = flow.step(state)

    assert new_state.sum() == pytest.approx(state.sum())


def test_markov_step_moves_mass_along_edges():
    G = nx.path_graph(5, create_using=nx.DiGraph)
    flow = MarkovChainFlow(G)
    state = flow.initial_state([0])

    new_state = flow.step(state)

    idx_0 = flow.node_order.index(0)
    idx_1 = flow.node_order.index(1)

    assert new_state[idx_0] == pytest.approx(0.0)
    assert new_state[idx_1] == pytest.approx(1.0)


def test_markov_step_supports_source_injection(directed_triangle):
    flow = MarkovChainFlow(directed_triangle)
    state = flow.initial_state([0])
    source = flow.initial_state([0])

    new_state = flow.step(state, source=source)

    assert new_state.sum() == pytest.approx(2.0)


def test_markov_apply_given_zero_time_returns_same_state(directed_triangle):
    flow = MarkovChainFlow(directed_triangle)
    state = flow.initial_state(list(directed_triangle.nodes)[:1])

    result = flow.apply(state, time=0)

    np.testing.assert_array_equal(result, state)


def test_markov_apply_one_step_equals_step(directed_triangle):
    flow = MarkovChainFlow(directed_triangle)
    state = flow.initial_state(list(directed_triangle.nodes)[:1])

    next_state_from_step = flow.step(state)
    next_state_from_apply = flow.apply(state, time=1)

    np.testing.assert_allclose(next_state_from_apply, next_state_from_step)


def test_markov_apply_converges_to_stationary_distribution_for_normal_markov_chain(  # noqa: E501
    undirected_triangle,
):
    flow = MarkovChainFlow(undirected_triangle)
    state = flow.initial_state(list(undirected_triangle.nodes)[:1])
    total_nodes = len(undirected_triangle.nodes)

    result = flow.apply(state, time=1000)

    stationary_distribution = np.full(total_nodes, 1.0 / total_nodes)
    np.testing.assert_allclose(result, stationary_distribution, atol=1e-6)


def test_markov_apply_without_heat_sources_conserves_total_mass(
    directed_triangle,
):
    flow = MarkovChainFlow(directed_triangle)
    state = flow.initial_state(list(directed_triangle.nodes)[:1])

    result = flow.apply(state, time=25)

    assert result.sum() == pytest.approx(1.0)


def test_markov_respects_edge_weights():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=3.0)
    G.add_edge(0, 2, weight=1.0)
    flow = MarkovChainFlow(G, weight='weight')
    state = flow.initial_state([0])

    new_state = flow.step(state)

    index_1 = flow.node_order.index(1)
    index_2 = flow.node_order.index(2)
    assert new_state[index_1] == pytest.approx(0.75)
    assert new_state[index_2] == pytest.approx(0.25)


def test_capacity_constrained_flow_satisfies_protocol(directed_triangle):
    nx.set_edge_attributes(directed_triangle, name='capacity', values=1.0)

    flow = CapacityConstrainedFlow(directed_triangle)

    assert isinstance(flow, Flow)


def test_capacity_constrained_flow_raises_on_nan_capacity():
    G = nx.DiGraph([(0, 1, {'capacity': np.nan})])

    with pytest.raises(ValueError, match='non-finite'):
        CapacityConstrainedFlow(G)


def test_capacity_constrained_flow_raises_on_infinite_capacity():
    G = nx.DiGraph([(0, 1, {'capacity': np.inf})])

    with pytest.raises(ValueError, match='non-finite'):
        CapacityConstrainedFlow(G)


def test_capacity_constrained_flow_raises_on_negative_capacity():
    G = nx.DiGraph([(0, 1, {'capacity': -1.0})])

    with pytest.raises(ValueError, match='negative'):
        CapacityConstrainedFlow(G)


def test_capacity_constrained_flow_step_changes_node_weights_based_on_states_and_capacities():
    G = nx.DiGraph([(0, 1, {'capacity': 0.3}), (1, 2, {'capacity': 1.8})])
    flow = CapacityConstrainedFlow(G)

    state = flow.initial_state([0, 1, 2], initial=1)
    new_state = flow.step(state)

    index_0 = flow.node_order.index(0)
    index_1 = flow.node_order.index(1)
    index_2 = flow.node_order.index(2)

    assert new_state[index_0] == pytest.approx(0.7)
    assert new_state[index_1] == pytest.approx(0.3)
    assert new_state[index_2] == pytest.approx(2.0)


def test_capacity_constrained_flow_capacity_limits_flow_through_edge():
    G = nx.DiGraph([(0, 1)])
    G.edges[0, 1]['capacity'] = 0.5
    flow = CapacityConstrainedFlow(G)
    index_0 = flow.node_order.index(0)
    index_1 = flow.node_order.index(1)
    state = flow.initial_state([0], initial=10.0)

    new_state = flow.step(state)

    assert new_state[index_1] == pytest.approx(0.5)
    assert new_state[index_0] == pytest.approx(9.5)


def test_capacity_constrained_flow_given_larger_capacity_than_occupancy_yields_full_departure():
    G = nx.DiGraph([(0, 1)])
    G.edges[0, 1]['capacity'] = 100.0
    flow = CapacityConstrainedFlow(G)
    index_0 = flow.node_order.index(0)
    index_1 = flow.node_order.index(1)
    state = flow.initial_state([0], initial=1.0)

    new_state = flow.step(state)

    assert new_state[index_0] == pytest.approx(0.0)
    assert new_state[index_1] == pytest.approx(1.0)


@hyp.given(
    G=random_digraph(
        edge_data=st.fixed_dictionaries(
            {'capacity': st.floats(min_value=0.01, max_value=10.0)}
        )
    ),
    initial=st.floats(min_value=0.1, max_value=100.0),
    steps=st.integers(min_value=1, max_value=50),
)
def test_capacity_constrained_flow_step_conserves_total_mass_without_source(
    G, initial, steps
):
    flow = CapacityConstrainedFlow(G)
    state = flow.initial_state(list(G.nodes)[:1], initial=initial)

    for _ in range(steps):
        state = flow.step(state)

    assert state.sum() == pytest.approx(initial, rel=1e-10)
    assert np.all(state >= -1e-15)


def test_capacity_constrained_flow_equals_markov_when_capacities_match_weights():
    G = nx.DiGraph(
        [
            (0, 1, {'capacity': 3.0}),
            (1, 2, {'capacity': 1.0}),
            (2, 0, {'capacity': 2.0}),
        ]
    )
    markov = MarkovChainFlow(G, weight='capacity')
    capacity = CapacityConstrainedFlow(G)

    initial_state_markov = markov.initial_state([0], initial=0.5)
    initial_state_capacity = capacity.initial_state([0], initial=0.5)
    updated_state_markov = markov.step(initial_state_markov)
    updated_state_capacity = capacity.step(initial_state_capacity)

    np.testing.assert_allclose(initial_state_markov, initial_state_capacity)
    np.testing.assert_allclose(
        updated_state_markov,
        updated_state_capacity,
        atol=1e-12,
    )
