"""Create heatmaps from NetworkX graphs."""

__version__ = '0.1.3'
__all__ = ['temperature_graph']

import itertools

from typing import Iterable, Optional, Union

import networkx as nx


def temperature_graph(
    G: nx.Graph,
    source_nodes: Iterable,
    depth_limit: Optional[int] = None,
    heat_increments: Union[Iterable, float] = 1,
    heat_key: str = 'heat'
) -> nx.Graph:
    """
    Calculate temperatures radiating from heat sources in a graph.

    Temperature values are initially set to 0 and then updated throughout `G` in
    a breadth-first manner beginning at each node in `source_nodes`.  For each
    source node `s`, the temperature of each edge `e` and its incident nodes in
    `G` are updated according to `heat_increments` and how many edges away they
    are from the source node `s`.  This process is repeated for every source
    node, with temperatures from multiple source nodes contibuting additively to
    the overall temperature of the nodes and edges in the graph.


    Parameters
    ----------
    G : networkx.Graph
        The graph from which to generate a heatmap.  A copy of the graph will be
        produced by default.

    source_nodes : Iterable
        The nodes serving as heat sources in `G`.

    depth_limit : Optional[int]
        The maximum number of edges away from a source node to update
        temperature values.  (Default: 0)

    heat_increments : Union[Iterable, float]
        A sequence whose `n`-th element gives, for each source node `s`, the
        amount to update the temperature of each node and edge that is `n`
        breadth-first layers away from `s`.  A constant value may also be
        provided to apply to all nodes and edges in the same connected component
        as each source node.  (Default: 1)

    heat_key : str
        The name of the node and edge attribute where temperature values will be
        stored in `T`.

    Returns
    -------
    T : networkx.Graph
        A copy of `G` in which every node and edge has its temperature stored in
        a `heat_key` attribute.
    """
    T = type(G)()
    T.add_nodes_from(G.nodes(), **{heat_key: 0})
    T.add_edges_from(G.edges(), **{heat_key: 0})

    try:
        heat_increments = iter(heat_increments)
    except TypeError:
        heat_increments = itertools.repeat(heat_increments)

    for source in source_nodes:
        visited_nodes = set()
        data_by_depth = itertools.islice(
            zip(_edge_bfs_by_depth(T, [source]), heat_increments),
            depth_limit)
        for edges_at_depth, heat_increment in data_by_depth:
            for edge in edges_at_depth:
                _update_edge_temperature(T, edge, heat_key, heat_increment)
                _update_node_temperatures(T, edge[:2], heat_key, heat_increment,
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
            current_group_sources = {target for _, target, *_ in current_group}
            current_group = [current_edge]

    yield current_group


def _update_edge_temperature(G, edge, heat_key, heat_increment):
    G.edges[edge][heat_key] += heat_increment


def _update_node_temperatures(G, nodes, heat_key, heat_increment, visited_nodes):
    for node in set(nodes).difference(visited_nodes):
        G.nodes[node][heat_key] += heat_increment
        visited_nodes.add(node)
