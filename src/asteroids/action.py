from enum import IntEnum


class Action(IntEnum):
    LEFT = 0
    NOOP = 1
    RIGHT = 2

    def to_step(self):
        return self.value - 1

    def update_position(self, position, width):
        new_position = position + self.to_step()
        if new_position < 0:
            return 0
        if new_position >= width:
            return width - 1
        return new_position
