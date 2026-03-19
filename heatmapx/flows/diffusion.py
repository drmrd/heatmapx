from flows import FlowBase


class DiffusionFlow(FlowBase):
    def step(self, state, source=None):
        return state

    def apply(self, state, time, source=None):
        return state
