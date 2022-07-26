from enum import IntEnum

import numpy as np


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

    def update_position(self, position, width):
        new_position = position + self.to_step()
        if new_position < 0:
            return 0
        if new_position >= width:
            return width - 1
        return new_position
