from dynamics import FlowDynamicBase


class MarkovChainDynamic(FlowDynamicBase):
    def step(self, state, source=None): return state

    def apply(self, state, time, source=None): return state