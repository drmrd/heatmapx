from functools import cached_property

import numpy as np
import networkx as nx
import scipy.sparse

from . import FlowBase


class MarkovChainFlow(FlowBase):
    def __init__(self, graph: nx.Graph, weight: str = 'weight'):
        super().__init__(graph)
        self._weight = weight
        self._validate_weights()

    @cached_property
    def transition_matrix(self):
        W = nx.to_scipy_sparse_array(
            self.graph, weight=self._weight, nodelist=self.node_order
        ).T.tocsr()

        d_out = np.asarray(W.sum(axis=0)).flatten()
        if np.any(np.isinf(d_out)):
            underflowed_nodes = [
                node for node, d in zip(self.node_order, d_out) if np.isinf(d)
            ]
            raise ValueError(
                'Failed to construct transition matrix. The following nodes '
                f'have infinite weighted out-degree: {underflowed_nodes}. '
                'Their edge weights are too large to sum without overflow.'
            )
        d_out_inv = np.zeros_like(d_out, dtype=float)
        with np.errstate(over='ignore'):
            np.divide(1.0, d_out, out=d_out_inv, where=d_out != 0)
        if np.any(np.isinf(d_out_inv)):
            underflowed_nodes = [
                node
                for node, d in zip(self.node_order, d_out_inv)
                if np.isinf(d)
            ]
            raise ValueError(
                'Failed to construct transition matrix. The following nodes '
                f'have infinite weighted out-degree: {underflowed_nodes}. '
                'Their edge weights are too small to invert without underflow.'
            )

        # P = W @ diag(d_out^{-1}); column-stochastic
        P = W.multiply(d_out_inv)

        # Sinks get a self-loop to remain absorbing
        is_sink = d_out == 0
        return P + scipy.sparse.diags(is_sink.astype(float))

    def step(self, state, source=None):
        updated_state = self.transition_matrix @ state
        if source is not None:
            updated_state += source
        return updated_state

    def apply(self, state, time, source=None):
        node_count = len(state)

        if source is None:
            step_map = self.transition_matrix
        else:
            source_column = source[:, np.newaxis]
            # The affine `state |-> P @ state + source` map lifted to a
            # linear map in homogeneous coordinates.
            step_map = scipy.sparse.bmat(
                [
                    [
                        self.transition_matrix,
                        scipy.sparse.csr_array(source_column),
                    ],
                    [
                        scipy.sparse.csr_array((1, node_count)),
                        scipy.sparse.eye(1),
                    ],
                ],
                format='csr',
            )
            state = np.append(state, 1.0)

        return (scipy.sparse.linalg.matrix_power(step_map, time) @ state)[
            :node_count
        ]

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
