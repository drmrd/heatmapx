__version__ = '0.1.0'
__all__ = ['temperature_graph']

from numbers import Real
from typing import Iterable

import networkx as nx


def temperature_graph(
    G: nx.Graph,
    source_nodes: Iterable,
    heat_increment: Real = 1,
    heat_key: str = 'heat'
) -> nx.Graph:
    T = type(G)()
    T.add_nodes_from(G.nodes(), **{heat_key: 0})
    T.add_edges_from(G.edges(), **{heat_key: 0})

    for source in source_nodes:
        visited_nodes = set()
        for edges_at_depth in _edge_bfs_by_depth(T, [source]):
            for edge in edges_at_depth:
                T.edges[edge][heat_key] += heat_increment
                for node in edge:
                    if node not in visited_nodes:
                        T.nodes[node][heat_key] += heat_increment
                        visited_nodes.add(node)
    return T


def _edge_bfs_by_depth(G, source_nodes, orientation=None):
    yield from _group_by_sources(
        nx.edge_bfs(G, source_nodes, orientation),
        set(source_nodes))


def _group_by_sources(edges_iterator, initial_sources):
    edges = iter(edges_iterator)

    current_group_sources = set(initial_sources)
    current_group = []

    for current_edge in edges:
        if current_edge[0] in current_group_sources:
            current_group.append(current_edge)
        else:
            yield current_group
            current_group_sources = {target for _, target in current_group}
            current_group = [current_edge]

    yield current_group


