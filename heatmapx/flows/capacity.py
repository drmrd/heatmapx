import warnings
from functools import cached_property

import networkx as nx
import numpy as np

from . import FlowBase


class CapacityConstrainedFlow(FlowBase):
    def __init__(self, graph: nx.Graph, capacity_attribute: str = 'capacity'):
        super().__init__(graph)
        self._capacity_attribute = capacity_attribute

        self._validate_weights()

    def step(self, state, source=None):
        effective_state = state if source is None else state + source

        # Normalized and unnormalized edgewise capacity matrices of the
        # graph
        P, C = self._capacity_matrices

        # Edgewise capacity-constrained flow matrix:
        #     F[i,j] = min(P[i,j] * state[j], C[i,j])
        F = P.multiply(effective_state).minimum(C)

        outflow = np.asarray(F.sum(axis=0)).flatten()
        inflow = np.asarray(F.sum(axis=1)).flatten()
        return effective_state - outflow + inflow

    def apply(self, state, time, source=None):
        for _ in range(time):
            state = self.step(state, source)
        return state

    def _validate_weights(self):
        has_nonfinite_edge_weight = any(
            not np.isfinite(data.get(self._capacity_attribute))
            for *_, data in self.graph.edges(keys=True, data=True)
            if self._capacity_attribute in data
        )
        if has_nonfinite_edge_weight:
            raise ValueError(
                'Edges exist in the provided graph with non-finite capacity '
                'attributes.'
            )

        has_negative_weight = any(
            data.get(self._capacity_attribute) < 0
            for *_, data in self.graph.edges(keys=True, data=True)
            if self._capacity_attribute in data
        )
        if has_negative_weight:
            raise ValueError(
                'Edges exist in the provided graph with negative capacity '
                'attributes.'
            )

        has_positive_weight = any(
            data.get(self._capacity_attribute) > 0
            for *_, data in self.graph.edges(keys=True, data=True)
            if self._capacity_attribute in data
        )
        if not has_positive_weight:
            warnings.warn(
                f'All {self._capacity_attribute} values are zero in this '
                'flow. At least one value must be positive for the flow to '
                "alter the graph's current state.",
                UserWarning,
            )

    @cached_property
    def _capacity_matrices(self):
        # Calculate the unnormalized capacities of each edge in the
        # graph
        C = nx.to_scipy_sparse_array(
            self.graph,
            weight=self._capacity_attribute,
            nodelist=self.node_order,
            dtype=np.float64,
        ).T.tocsr()

        # Normalize based on the out-capacity of each node
        d_out = np.asarray(C.sum(axis=0)).flatten()
        d_out_inv = np.zeros_like(d_out, dtype=float)
        np.divide(1.0, d_out, out=d_out_inv, where=d_out != 0)

        # P = C @ diag(d_out^{-1}); column-stochastic
        P = C.multiply(d_out_inv)
        return P, C
