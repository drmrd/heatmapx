import numpy as np
import networkx as nx
import scipy.sparse

from dynamics import FlowDynamicBase


class MarkovChainDynamic(FlowDynamicBase):
    def __init__(self, graph: nx.Graph, departure_rate: float = 1.0):
        super().__init__(graph)
        self._departure_rate = departure_rate

    @property
    def departure_rate(self):
        return self._departure_rate

    @property
    def transition_matrix(self):
        W = nx.to_scipy_sparse_array(self.graph, weight='heat')
        d_out = np.asarray(W.sum(axis=1)).flatten()
        d_out_inv = np.zeros_like(d_out, dtype=float)
        np.divide(1.0, d_out, out=d_out_inv, where=d_out != 0)

        # P = alpha * (D_out^{-1} W)^T; column-stochastic
        P = (
            self.departure_rate * scipy.sparse.diags(d_out_inv) @ W
        ).T.tocsr()

        # Self-loop correction: (1 - departure_rate) for non-sinks, 1 for sinks
        return P + scipy.sparse.diags(
            np.where(d_out > 0, 1.0 - self.departure_rate, 1.0)
        )

    def step(self, state, source=None):
        return self.transition_matrix @ state

    def apply(self, state, time, source=None): return state