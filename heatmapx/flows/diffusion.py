import networkx as nx
import numpy as np
import scipy

from flows import FlowBase


class DiffusionFlow(FlowBase):
    @property
    def laplacian(self):
        W = nx.to_scipy_sparse_array(
            self.graph, weight='weight', nodelist=self.node_order
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
        return state

    def apply(self, state, time, source=None):
        return state
