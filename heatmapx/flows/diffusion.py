from functools import cached_property

import networkx as nx
import numpy as np
import scipy

from flows import FlowBase


class DiffusionFlow(FlowBase):
    def __init__(self, graph: nx.Graph, weight: str = 'weight'):
        super().__init__(graph)
        self._weight = weight

    @cached_property
    def laplacian(self):
        W = nx.to_scipy_sparse_array(
            self.graph, weight=self._weight, nodelist=self.node_order
        ).T.tocsr()

        d_out = np.asarray(W.sum(axis=0)).flatten()
        d_out_inv = np.zeros_like(d_out, dtype=np.float64)
        np.divide(1.0, d_out, out=d_out_inv, where=d_out != 0)

        # P = W @ diag(d_out^{-1}); column-stochastic
        P = W.multiply(d_out_inv)

        # Sinks get a self-loop to remain absorbing
        is_sink = d_out == 0
        transition_matrix = P + scipy.sparse.diags(is_sink.astype(np.float64))

        return (
            scipy.sparse.eye(len(self.graph), dtype=np.float64, format='csr')
            - transition_matrix
        )

    def step(self, state, source=None):
        return self.apply(state, time=1.0, source=source)

    def apply(self, state, time, source=None):
        node_count = len(state)

        if source is None:
            step_map = -self.laplacian
        else:
            source_column = source[:, np.newaxis]
            # The affine `state |-> exp(-t * L) @ state + source` map
            # lifted to a linear map in homogeneous coordinates.
            step_map = scipy.sparse.bmat(
                [
                    [
                        -self.laplacian,
                        scipy.sparse.csr_array(source_column),
                    ],
                    [
                        scipy.sparse.csr_array((1, node_count)),
                        None,
                    ],
                ],
                format='csr',
            )
            state = np.append(state, 1.0)

        return scipy.sparse.linalg.expm_multiply(step_map * time, state)[
            :node_count
        ]
