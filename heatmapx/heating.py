import networkx as nx
import numpy as np
import scipy.linalg


def heat_graph(G: nx.Graph, source, time) -> nx.Graph:
    G_directed = G.to_directed()
    A = nx.adjacency_matrix(G_directed, weight=None).todense().transpose()
    D = np.diag([out_degree for _, out_degree in G_directed.out_degree])

    D_inv_breve = 1 / D
    D_inv_breve[D_inv_breve == np.inf] = 0

    L = D - A
    laplacian = L @ D_inv_breve

    source_encoded = np.array([
        1 if node == source else 0
        for node in G_directed.nodes
    ])

    heat_coefficients = scipy.linalg.expm(
        -time * laplacian
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
