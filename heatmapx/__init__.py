__version__ = '0.1.0'

from typing import Iterable

import networkx as nx


def temperature_graph(
    G: nx.Graph,
    source_nodes: Iterable,
) -> nx.Graph:
    T = type(G)()
    return T
