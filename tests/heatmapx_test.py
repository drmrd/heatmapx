import networkx as nx
import pytest

import heatmapx as hx


def test_given_a_graph_and_source_nodes_returns_a_new_graph():
    G = nx.Graph()
    G_temperatures = hx.temperature_graph(G, sources=[])
    assert isinstance(G_temperatures, nx.Graph)
    assert G_temperatures != G


@pytest.mark.parametrize(
    'input_class',
    [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
)
def test_output_type_matches_its_input(input_class):
    G = input_class()
    G_temperatures = hx.temperature_graph(G, sources=[])
    assert isinstance(G_temperatures, input_class)


def test_output_has_same_graph_structure_as_input():
    G = nx.Graph()
    G_temperatures = hx.temperature_graph(G, sources=[])
    assert set(G_temperatures.nodes()) == set(G.nodes())
    assert set(G_temperatures.edges()) == set(G.edges())


def test_all_output_nodes_and_edges_store_heat_data():
    heat_key = 'heat goes here!'

    cyclic_graph = nx.cycle_graph(3)
    cyclic_temperature_graph = hx.temperature_graph(
        cyclic_graph, sources=[], key=heat_key)

    for node in cyclic_graph.nodes():
        assert heat_key in cyclic_temperature_graph.nodes[node]
    for source, target in cyclic_temperature_graph.edges():
        assert heat_key in cyclic_temperature_graph.edges[source, target]


def test_the_default_heat_key_is_heat():
    cyclic_graph = nx.cycle_graph(3)
    cyclic_temperature_graph = hx.temperature_graph(cyclic_graph, sources=[])
    assert 'heat' in cyclic_temperature_graph.nodes[0]


@pytest.mark.parametrize(
    'graph_class',
    [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
)
def test_heat_data_is_updated_throughout_graph(graph_class):
    graph = nx.complete_graph(7, create_using=graph_class)
    temperature_graph = hx.temperature_graph(graph, sources=[0])

    for node in graph.nodes:
        assert temperature_graph.nodes[node]['heat'] == 1
    for edge in graph.edges:
        assert temperature_graph.edges[edge]['heat'] == 1


@pytest.mark.parametrize(
    'graph_class',
    [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
)
@pytest.mark.parametrize(
    'node_labels',
    [[1, 3, 5, 7],
     ['n1', 'n2', 'n3', 'n4']]
)
def test_supports_common_graph_and_node_label_types(graph_class, node_labels):
    graph = graph_class(list(zip(node_labels[:-1], node_labels[1:])))
    temperature_graph = hx.temperature_graph(graph, sources=[node_labels[0]])

    for node in graph.nodes:
        assert temperature_graph.nodes[node]['heat'] == 1
    for edge in graph.edges:
        assert temperature_graph.edges[edge]['heat'] == 1


@pytest.mark.parametrize(
    'graph_class',
    [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
)
def test_components_disconnected_from_sources_are_assigned_coldest_temperature(graph_class):
    graph: nx.Graph = graph_class([(0, 1), (1, 2), (3, 4)])

    temperature_graph = hx.temperature_graph(graph, [0], increments=range(1, 4))

    source_connected_component_max_temperature = max(
        *(heat for node, heat in temperature_graph.nodes(data='heat') if node <= 2),
        *(heat for *_, heat in temperature_graph.edges({0, 1, 2}, data='heat'))
    )

    assert {
        temperature_graph.nodes[3]['heat'],
        temperature_graph.nodes[4]['heat'],
        temperature_graph.get_edge_data(3, 4, 0)['heat']
    } == {source_connected_component_max_temperature}


def test_heat_from_multiple_sources_accumulates_additively():
    square_graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
    temperature_graph = hx.temperature_graph(square_graph, [1, 2])

    for node in square_graph.nodes:
        assert temperature_graph.nodes[node]['heat'] == 2
    for edge in square_graph.edges:
        assert temperature_graph.edges[edge]['heat'] == 2


def test_heat_distribution_respects_edge_directedness():
    square_graph = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    temperature_graph = hx.temperature_graph(square_graph, [0],
                                             increments=range(3))
    assert temperature_graph.nodes[1]['heat'] == 1
    assert temperature_graph.nodes[2]['heat'] == 2


@pytest.mark.parametrize(
    'graph_class',
    [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
)
def test_accepts_iterable_of_depth_specific_heat_increments(graph_class):
    square_graph = graph_class([(0, 1), (0, 2), (1, 3), (2, 3)])

    hx.temperature_graph(
        square_graph,
        sources=[0],
        increments=[1, 0.5, 0.25]
    )


@pytest.mark.parametrize('graph_class', [nx.Graph, nx.DiGraph])
def test_heat_increments_update_for_edge_targets(graph_class):
    single_edge_graph = graph_class([('A', 'B')])
    temperature_graph = hx.temperature_graph(
        single_edge_graph, sources=['A'], increments=[1, 0.5]
    )
    assert temperature_graph.nodes['A']['heat'] == 1
    assert temperature_graph.edges['A', 'B']['heat'] == 1
    assert temperature_graph.nodes['B']['heat'] == 0.5


@pytest.mark.parametrize(
    'graph_class',
    [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
)
def test_can_limit_distance_heat_spreads_from_heat_sources(graph_class):
    square_graph = graph_class([(0, 1), (0, 2), (1, 3), (2, 3)])

    heat_source = 0
    temperature_graph = hx.temperature_graph(
        square_graph,
        sources=[heat_source],
        max_depth=1
    )

    heat_source_and_neighbors = {heat_source}.union(
        set(temperature_graph.neighbors(heat_source)))
    incident_edges = set(graph_edges(temperature_graph, heat_source))

    for neighbor in heat_source_and_neighbors:
        assert temperature_graph.nodes[neighbor]['heat'] > 0
    for edge in incident_edges:
        assert temperature_graph.edges[edge]['heat'] > 0

    for node in set(temperature_graph.nodes) - heat_source_and_neighbors:
        assert temperature_graph.nodes[node]['heat'] == 0
    for edge in set(graph_edges(temperature_graph)) - incident_edges:
        assert temperature_graph.edges[edge]['heat'] == 0


def test_heat_increments_can_be_weighted_by_a_graph_attribute():
    single_edge_graph = nx.DiGraph([('A', 'B')])
    single_edge_graph.nodes['A']['some_attribute'] = 2
    single_edge_graph.edges['A', 'B']['some_attribute'] = 3
    single_edge_graph.nodes['B']['some_attribute'] = 5

    temperature_graph = hx.temperature_graph(
        single_edge_graph, sources=['A'], increments=[1, 0.5],
        weight='some_attribute'
    )
    assert temperature_graph.nodes['A']['heat'] == 1 * 2
    assert temperature_graph.edges['A', 'B']['heat'] == 1 * 3
    assert temperature_graph.nodes['B']['heat'] == 0.5 * 5


def test_unreachable_nodes_and_edges_are_assigned_minimum_temperature_value():
    G = nx.path_graph(5, create_using=nx.DiGraph)
    G.add_edge(2, 4)

    source = 2
    unreachable_nodes = [0, 1]
    unreachable_edges = [(0, 1), (1, 2)]

    H = hx.temperature_graph(G, [source])
    coldest_temperature = max(
        max(heat for *_, heat in H.edges.data('heat')),
        max(heat for _, heat in H.nodes.data('heat'))
    )
    assert coldest_temperature == 1

    for unreachable_node in unreachable_nodes:
        assert H.nodes[unreachable_node]['heat'] == coldest_temperature
    for unreachable_edge in unreachable_edges:
        assert H.edges[unreachable_edge]['heat'] == coldest_temperature


def graph_edges(graph, nbunch=None, data=False):
    edges_kwargs = {'nbunch': nbunch, 'data': data}
    if isinstance(graph, nx.MultiGraph):
        edges_kwargs['keys'] = True
    return graph.edges(**edges_kwargs)