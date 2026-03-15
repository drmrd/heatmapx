import networkx as nx
import numpy as np

from . import FlowBase


class CapacityConstrainedFlow(FlowBase):
    def __init__(self, graph: nx.Graph):
        super().__init__(graph)

    def step(self, state, source=None):
        # Calculate the unnormalized capacities of each edge in the graph
        C = nx.to_scipy_sparse_array(
            self.graph, weight='capacity', nodelist=self.node_order,
            dtype=np.float64,
        ).T.tocsr()

        # Normalize based on the out-capacity of each node
        d_out = np.asarray(C.sum(axis=0)).flatten()
        d_out_inv = np.zeros_like(d_out, dtype=float)
        np.divide(1.0, d_out, out=d_out_inv, where=d_out != 0)

        # P = C @ diag(d_out^{-1}); column-stochastic
        P = C.multiply(d_out_inv)

        # Edgewise capacity-constrained flow matrix:
        #     F[i,j] = min(P[i,j] * state[j], C[i,j])
        F = P.multiply(state).minimum(C)

        outflow = np.asarray(F.sum(axis=0)).flatten()
        inflow = np.asarray(F.sum(axis=1)).flatten()
        return state - outflow + inflow

    def apply(self, state, time, source=None):
        return state
