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
