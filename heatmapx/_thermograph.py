import itertools
from typing import Iterable, Iterator, Optional, Union

import networkx as nx


def temperature_graph(
    G: nx.Graph,
    sources: Iterable,
    max_depth: Optional[int] = None,
    increments: Union[Iterable[Union[int, float]], int, float] = 1.0,
    weight: Optional[str] = None,
    key: Optional[str] = 'heat'
) -> nx.Graph:
    """
    Calculate temperatures radiating from heat sources in a graph.

    Temperature values are initially set to 0 and then updated throughout
    `G` in a breadth-first manner beginning at each node in `sources`.
    For each source node `s`, the temperature of each edge `e` and its
    incident nodes in `G` are updated according to `increments` and how
    many edges away they are from the source node `s`. This process is
    repeated for every source node, with temperatures from multiple source
    nodes contributing additively to the overall temperature of the nodes
    and edges in the graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph from which to generate a heatmap.

    sources : iterable
        The nodes serving as heat sources in `G`.

    max_depth : int, optional
        The number of breadth-first search layers to traverse while
        updating node and edge temperatures. (Equivalently, the maximum
        graph distance away from a source node at which to update
        temperatures.) If left unspecified, all nodes and edges reachable
        from a source node will be updated.

    increments : iterable of float or float, default 1.0
        A sequence whose `n`-th element gives, for each source node `s`,
        the amount to update the temperature of each node and edge that is
        `n` breadth-first layers away from `s`. A constant value may also
        be provided to apply to all nodes and edges in the same connected
        component as each source node.

        If the provided iterable is exhausted before `temperature_graph`
        finishes calculating temperatures for graph elements, the final
        increment in the iterable will be used for the remainder of the
        graph.

    weight : str, optional
        A node and edge attribute that should be used to multiplicatively
        scale heat increments. Heat increments are not scaled by a weight
        attribute by default.

    key : str, default 'heat'
        The name of the node and edge attribute where temperature values
        will be stored in `T`.

    Returns
    -------
    networkx.Graph
        A deep copy of `G` with a temperature assigned to the `key`
        attribute of each node and edge.
    """
    T = type(G)()
    T.add_nodes_from(G.nodes(), **{key: 0})
    T.add_edges_from(G.edges(), **{key: 0})

    if weight is not None:
        nx.set_node_attributes(T, nx.get_node_attributes(G, weight), weight)
        nx.set_edge_attributes(T, nx.get_edge_attributes(G, weight), weight)

    try:
        increments, increments_tester = itertools.tee(increments)
        _validate_iterator_is_nonempty(increments_tester)
        increments = _repeat_last(increments)
    except TypeError:
        increments = itertools.repeat(increments)
    except ValueError:
        raise ValueError('Provided increments iterable must be nonempty.')

    for source in sources:
        visited_nodes = set()
        data_by_depth = itertools.islice(
            zip(
                _edge_bfs_by_depth(T, [source]),
                _consecutive_pairs(increments)
            ),
            max_depth
        )
        for edges_at_depth, (increment, next_increment) in data_by_depth:
            for edge in edges_at_depth:
                _update_edge_temperature(T, edge, key, increment,
                                         next_increment, weight)
                _update_incident_node_temperatures(T, edge[:2], key, increment,
                                                   next_increment, weight,
                                                   visited_nodes)
    _update_temperatures_of_unreachable_nodes_and_edges(T, sources, key)
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


def _consecutive_pairs(iterable):
    first_items, second_items = itertools.tee(iterable)
    next(second_items, None)
    return itertools.zip_longest(first_items, second_items, fillvalue=0)


def _update_edge_temperature(G, edge, key, increment, next_increment,
                             weight):
    G.edges[edge][key] += G.edges[edge].get(weight, 1) * increment


def _update_incident_node_temperatures(G, nodes, key, increment,
                                       next_increment, weight, visited_nodes):
    source, target = nodes
    if source not in visited_nodes:
        G.nodes[source][key] += G.nodes[source].get(weight, 1) * increment
    if target not in visited_nodes:
        G.nodes[target][key] += G.nodes[target].get(weight, 1) * next_increment
    visited_nodes.update(nodes)


def _update_temperatures_of_unreachable_nodes_and_edges(T, sources, key):
    unreachable_nodes = set(T) - set(sources) - set().union(*(
        nx.descendants(T, source) for source in sources
    ))
    coldest_temperature = max(
        max((heat for *_, heat in T.edges.data(key)), default=0),
        max((heat for _, heat in T.nodes.data(key)), default=0)
    )

    for unreachable_node in unreachable_nodes:
        T.nodes[unreachable_node][key] = coldest_temperature

    edges_iterator_kwargs = {}
    if isinstance(T, nx.MultiGraph):
        edges_iterator_kwargs['keys'] = True
    for edge in T.edges(unreachable_nodes, **edges_iterator_kwargs):
        T.edges[edge][key] = coldest_temperature


def _validate_iterator_is_nonempty(iterator: Iterator):
    empty_iterator_flag = object()
    item = next(iterator, empty_iterator_flag)
    if item is empty_iterator_flag:
        raise ValueError('Provided iterator must be nonempty.')


def _repeat_last(iterable: Iterable):
    for item in iterable:
        yield item
    yield from itertools.repeat(item)