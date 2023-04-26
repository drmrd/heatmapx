import networkx as nx
import numpy as np
import scipy.linalg


def heat_graph(G: nx.Graph, sources, time) -> nx.Graph:
    G_directed = G.to_directed()
    A = nx.to_scipy_sparse_array(G_directed, weight=None)
    D = np.diag(np.asarray(A.sum(axis=1)))

    D_out_inv = 1 / D
    D_out_inv[D_out_inv == np.inf] = 0

    L = D - A
    L_out = D_out_inv @ L

    sources_indicator = scipy.sparse.csr_matrix(
        np.isin(
            G_directed.nodes,
            list(sources)
        )
    )

    heat_coefficients = sources_indicator @ scipy.linalg.expm(
        -time * L_out
    )

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
