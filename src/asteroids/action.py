from enum import IntEnum

import numpy as np

from asteroids.edge_policy import EdgePolicy


class Action(IntEnum):
    LEFT = 0
    NOOP = 1
    RIGHT = 2

    def to_vector(self):
        vec = np.zeros(shape=len(Action))
        vec[self.value] = 1
        return vec

    def to_step(self):
        return self.value - 1

    def update_position(self, position: int, width: int, edge_policy: EdgePolicy):
        return edge_policy.convert_position(
            position=position + self.to_step(), width=width
        )
