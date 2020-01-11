import heatmapx as hx
import networkx as nx


class TestHeatmapXPackaging:
    def test_version(self):
        assert hx.__version__ == '0.1.0'


class TestTemperatureGraph:
    def test_given_a_graph_and_source_nodes_returns_a_new_graph(self):
        G = nx.Graph()
        G_temperatures = hx.temperature_graph(G, source_nodes=[])
        assert isinstance(G_temperatures, nx.Graph)
        assert G_temperatures != G
