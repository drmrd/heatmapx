import networkx as nx
import pytest

import heatmapx as hx


class TestTemperatureGraph:
    def test_given_a_graph_and_source_nodes_returns_a_new_graph(self):
        G = nx.Graph()
        G_temperatures = hx.temperature_graph(G, sources=[])
        assert isinstance(G_temperatures, nx.Graph)
        assert G_temperatures != G

    @pytest.mark.parametrize(
        'input_class',
        [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
    )
    def test_output_type_matches_its_input(self, input_class):
        G = input_class()
        G_temperatures = hx.temperature_graph(G, sources=[])
        assert isinstance(G_temperatures, input_class)

    def test_output_has_same_graph_structure_as_input(self):
        G = nx.Graph()
        G_temperatures = hx.temperature_graph(G, sources=[])
        assert set(G_temperatures.nodes()) == set(G.nodes())
        assert set(G_temperatures.edges()) == set(G.edges())

    def test_all_output_nodes_and_edges_store_heat_data(self):
        heat_key = 'heat goes here!'

        cyclic_graph = nx.cycle_graph(3)
        cyclic_temperature_graph = hx.temperature_graph(
            cyclic_graph, sources=[], key=heat_key)

        for node in cyclic_graph.nodes():
            assert heat_key in cyclic_temperature_graph.nodes[node]
        for source, target in cyclic_temperature_graph.edges():
            assert heat_key in cyclic_temperature_graph.edges[source, target]

    def test_the_default_heat_key_is_heat(self):
        cyclic_graph = nx.cycle_graph(3)
        cyclic_temperature_graph = hx.temperature_graph(cyclic_graph, sources=[])
        assert 'heat' in cyclic_temperature_graph.nodes[0]

    @pytest.mark.parametrize(
        'graph_class',
        [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
    )
    def test_heat_data_is_updated_throughout_graph(self, graph_class):
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
    def test_supports_common_graph_and_node_label_types(self, graph_class, node_labels):
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
    def test_heat_data_is_only_updated_in_connected_components_of_source_nodes(self, graph_class):
        graph: nx.Graph = graph_class([(0, 1), (1, 2), (2, 0),
                                       (3, 4), (4, 5), (5, 3)])
        graph.add_nodes_from([6, 7, 8])

        temperature_graph = hx.temperature_graph(graph, [0])

        for node in range(3):
            assert temperature_graph.nodes[node]['heat'] == 1
        for node in range(3, 9):
            assert temperature_graph.nodes[node]['heat'] == 0

    def test_heat_from_multiple_sources_accumulates_additively(self):
        square_graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        temperature_graph = hx.temperature_graph(square_graph, [1, 2])

        for node in square_graph.nodes:
            assert temperature_graph.nodes[node]['heat'] == 2
        for edge in square_graph.edges:
            assert temperature_graph.edges[edge]['heat'] == 2

    def test_heat_distribution_respects_edge_directedness(self):
        square_graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
        temperature_graph = hx.temperature_graph(square_graph, [1, 2])

        assert temperature_graph.nodes[0]['heat'] == 0
        assert temperature_graph.edges[0, 1]['heat'] == 0
        assert temperature_graph.edges[0, 2]['heat'] == 0

        assert temperature_graph.nodes[1]['heat'] == 1
        assert temperature_graph.nodes[2]['heat'] == 1
        assert temperature_graph.edges[1, 3]['heat'] == 1
        assert temperature_graph.edges[2, 3]['heat'] == 1

        assert temperature_graph.nodes[3]['heat'] == 2

    @pytest.mark.parametrize(
        'graph_class',
        [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph]
    )
    def test_accepts_iterable_of_depth_specific_heat_increments(self, graph_class):
        square_graph = graph_class([(0, 1), (0, 2), (1, 3), (2, 3)])

        hx.temperature_graph(
            square_graph,
            sources=[0],
            increments=[1, 0.5, 0.25]
        )

    @pytest.mark.parametrize('graph_class', [nx.Graph, nx.DiGraph])
    def test_heat_increments_update_for_edge_targets(self, graph_class):
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
    def test_can_limit_distance_heat_spreads_from_heat_sources(self, graph_class):
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

    def test_heat_increments_can_be_weighted_by_a_graph_attribute(self):
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


def graph_edges(graph, nbunch=None, data=False):
    edges_kwargs = {'nbunch': nbunch, 'data': data}
    if isinstance(graph, nx.MultiGraph):
        edges_kwargs['keys'] = True
    return graph.edges(**edges_kwargs)