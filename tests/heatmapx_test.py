import networkx as nx
import pytest

import heatmapx as hx


class TestHeatmapXPackaging:
    def test_version(self):
        assert hx.__version__ == '0.1.0'


class TestTemperatureGraph:
    def test_given_a_graph_and_source_nodes_returns_a_new_graph(self):
        G = nx.Graph()
        G_temperatures = hx.temperature_graph(G, source_nodes=[])
        assert isinstance(G_temperatures, nx.Graph)
        assert G_temperatures != G

    @pytest.mark.parametrize(
        'input_class', [nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.OrderedDiGraph])
    def test_output_type_matches_its_input(self, input_class):
        G = input_class()
        G_temperatures = hx.temperature_graph(G, source_nodes=[])
        assert isinstance(G_temperatures, input_class)

    def test_output_has_same_graph_structure_as_input(self):
        G = nx.Graph()
        G_temperatures = hx.temperature_graph(G, source_nodes=[])
        assert set(G_temperatures.nodes()) == set(G.nodes())
        assert set(G_temperatures.edges()) == set(G.edges())

    def test_all_output_nodes_and_edges_store_heat_data(self):
        heat_key = 'heat goes here!'

        cyclic_graph = nx.cycle_graph(3)
        cyclic_temperature_graph = hx.temperature_graph(
            cyclic_graph, source_nodes=[], heat_key=heat_key)

        for node in cyclic_graph.nodes():
            assert heat_key in cyclic_temperature_graph.nodes[node]
        for source, target in cyclic_temperature_graph.edges():
            assert heat_key in cyclic_temperature_graph.edges[source, target]

    def test_the_default_heat_key_is_heat(self):
        cyclic_graph = nx.cycle_graph(3)
        cyclic_temperature_graph = hx.temperature_graph(cyclic_graph, source_nodes=[])
        assert 'heat' in cyclic_temperature_graph.nodes[0]

    @pytest.mark.parametrize(
        'graph',
        [nx.cycle_graph(3),
         nx.cycle_graph(5, create_using=nx.DiGraph),
         nx.karate_club_graph()]
    )
    def test_heat_data_is_updated_throughout_graph(self, graph):
        graph = nx.cycle_graph(3)
        temperature_graph = hx.temperature_graph(graph, source_nodes=[0])

        for node in graph.nodes:
            assert temperature_graph.nodes[node]['heat'] == 1
        for edge in graph.edges:
            assert temperature_graph.edges[edge]['heat'] == 1

    def test_heat_data_is_only_updated_in_connected_components_of_source_nodes(self):
        graph = nx.Graph([(0, 1), (1, 2), (2, 0),
                          (3, 4), (4, 5), (5, 3)])
        graph.add_nodes_from([6, 7, 8])

        temperature_graph = hx.temperature_graph(graph, [0])

        for node in range(3):
            assert temperature_graph.nodes[node]['heat'] == 1
        for node in range(3, 9):
            assert temperature_graph.nodes[node]['heat'] == 0

    def test_heat_from_multiple_sources_accumulates_additively(self):
        square_graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
        temperature_graph = hx.temperature_graph(square_graph, [0, 3])

        for node in square_graph.nodes:
            assert temperature_graph.nodes[node]['heat'] == 2
        for edge in square_graph.edges:
            assert temperature_graph.edges[edge]['heat'] == 2

    def test_accepts_iterable_of_depth_specific_heat_increments(self):
        square_graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])

        heat_source = 0
        temperature_graph = hx.temperature_graph(
            square_graph,
            source_nodes=[heat_source],
            heat_increments=[1, 0.5]
        )

        heat_source_and_neighbors = {heat_source}.union(
            set(temperature_graph.neighbors(heat_source)))
        incident_edges = set(temperature_graph.edges(heat_source))

        for neighbor in heat_source_and_neighbors:
            assert temperature_graph.nodes[neighbor]['heat'] == 1
        for edge in incident_edges:
            assert temperature_graph.edges[edge]['heat'] == 1

        for node in set(temperature_graph.nodes) - heat_source_and_neighbors:
            assert temperature_graph.nodes[node]['heat'] == 0.5
        for edge in set(temperature_graph.edges) - incident_edges:
            assert temperature_graph.edges[edge]['heat'] == 0.5

    def test_can_limit_distance_heat_spreads_from_heat_sources(self):
        square_graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])

        heat_source = 0
        temperature_graph = hx.temperature_graph(
            square_graph,
            source_nodes=[heat_source],
            heat_increments=[1, 0.5],
            depth_limit=1
        )

        heat_source_and_neighbors = {heat_source}.union(
            set(temperature_graph.neighbors(heat_source)))
        incident_edges = set(temperature_graph.edges(heat_source))

        for neighbor in heat_source_and_neighbors:
            assert temperature_graph.nodes[neighbor]['heat'] == 1
        for edge in incident_edges:
            assert temperature_graph.edges[edge]['heat'] == 1

        for node in set(temperature_graph.nodes) - heat_source_and_neighbors:
            assert temperature_graph.nodes[node]['heat'] == 0
        for edge in set(temperature_graph.edges) - incident_edges:
            assert temperature_graph.edges[edge]['heat'] == 0
