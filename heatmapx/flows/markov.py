import numpy as np
import networkx as nx
import scipy.sparse

from . import FlowBase


class MarkovChainFlow(FlowBase):
    def __init__(self, graph: nx.Graph, weight: str = 'weight'):
        super().__init__(graph)
        self._weight = weight
        self._validate_weights()

    @property
    def transition_matrix(self):
        W = nx.to_scipy_sparse_array(self.graph, weight=self._weight)

        d_out = np.asarray(W.sum(axis=1)).flatten()
        if np.any(np.isinf(d_out)):
            overflowed_nodes = [
                node for node, d in zip(self.node_order, d_out) if np.isinf(d)
            ]
            raise ValueError(
                ' '.join(
                    [
                        'Failed to construct transition matrix. The following nodes '
                        f'have infinite weighted out-degree: {overflowed_nodes}. Their',
                        'edge weights are too large to sum without overflow.',
                    ]
                )
            )

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

    def _validate_weights(self):
        has_nonfinite_edge_weight = any(
            not np.isfinite(data.get(self._weight))
            for *_, data in self.graph.edges(keys=True, data=True)
            if self._weight in data
        )
        if has_nonfinite_edge_weight:
            raise ValueError(
                f'Edges exist in the provided graph with non-finite '
                f'{self._weight} attributes.'
            )