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
