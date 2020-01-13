__version__ = '0.1.0'
__all__ = ['temperature_graph']

import itertools

from typing import Iterable, Optional, Union

import networkx as nx


def temperature_graph(
    G: nx.Graph,
    source_nodes: Iterable,
    heat_increments: Union[Iterable, float] = 1,
    heat_key: str = 'heat'
) -> nx.Graph:
    T = type(G)()
    T.add_nodes_from(G.nodes(), **{heat_key: 0})
    T.add_edges_from(G.edges(), **{heat_key: 0})

    try:
        heat_increments = iter(heat_increments)
    except TypeError:
        heat_increments = itertools.repeat(heat_increments)

    for source in source_nodes:
        visited_nodes = set()
        data_by_depth = zip(_edge_bfs_by_depth(T, [source]), heat_increments)
        for edges_at_depth, heat_increment_at_depth in data_by_depth:
            for edge in edges_at_depth:
                _update_temperature(T, edge, heat_key, heat_increment_at_depth,
                                    visited_nodes)
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


def _update_temperature(G, edge, heat_key, heat_increment, visited_nodes):
    G.edges[edge][heat_key] += heat_increment
    for node in edge:
        if node not in visited_nodes:
            G.nodes[node][heat_key] += heat_increment
            visited_nodes.add(node)
