import networkx as nx
import numpy as np
import scipy.linalg


def heat_graph(G: nx.Graph, source, time) -> nx.Graph:
    G_directed = G.to_directed()
    A = nx.adjacency_matrix(G_directed, weight=None).todense().transpose()
    D = np.diag(np.asarray(A.sum(axis=0))[0])

    D_out_inv = 1 / D
    D_out_inv[D_out_inv == np.inf] = 0

    L = D - A
    L_out = L @ D_out_inv

    source_encoded = np.array([
        1 if node == source else 0
        for node in G_directed.nodes
    ])

    heat_coefficients = scipy.linalg.expm(
        -time * L_out
    ) @ source_encoded

    nx.set_node_attributes(
        G_directed,
        {
            node: heat
            for node, heat in zip(
                G_directed.nodes,
                heat_coefficients
            )
        },
        'heat'
    )
    return G_directed
