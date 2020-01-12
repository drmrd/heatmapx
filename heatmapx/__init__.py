__version__ = '0.1.0'

from typing import Iterable

import networkx as nx


def temperature_graph(
    G: nx.Graph,
    source_nodes: Iterable,
    heat_key: str = 'heat'
) -> nx.Graph:
    T = type(G)()
    T.add_nodes_from(G.nodes(), **{heat_key: 0})
    T.add_edges_from(G.edges(), **{heat_key: 0})

    return T
