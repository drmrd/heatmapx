import numpy as np
import networkx as nx
import scipy.sparse

from . import FlowDynamicBase


class MarkovChainDynamic(FlowDynamicBase):
    def __init__(self, graph: nx.Graph, weight: str = 'weight'):
        super().__init__(graph)
        self._weight = weight

    @property
    def transition_matrix(self):
        W = nx.to_scipy_sparse_array(self.graph, weight=self._weight)
        d_out = np.asarray(W.sum(axis=1)).flatten()
        d_out_inv = np.zeros_like(d_out, dtype=float)
        np.divide(1.0, d_out, out=d_out_inv, where=d_out != 0)

        # (D_out^{-1} W)^T; column-stochastic
        P = (scipy.sparse.diags(d_out_inv) @ W).T.tocsr()

        # Sinks get a self-loop to remain absorbing
        is_sink = d_out == 0
        return P + scipy.sparse.diags(is_sink.astype(float))

    def step(self, state, source=None):
        updated_state = self.transition_matrix @ state
        if source is not None:
            updated_state += source
        return updated_state

    def apply(self, state, time, source=None):
        for _ in range(time):
            state = self.step(state, source)
        return state
