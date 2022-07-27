from enum import IntEnum


class EdgePolicy(IntEnum):
    WALL = 0
    FALL = 1
    CIRCULAR = 2

    @classmethod
    def names(cls):
        return [e.name for e in cls]

    def convert_position(self, position: int, width: int):
        if self == EdgePolicy.WALL:
            return self.convert_wall_position(position, width)
        if self == EdgePolicy.CIRCULAR:
            return self.convert_circular_position(position, width)
        return self.convert_fall_position(position, width)

    @classmethod
    def convert_wall_position(cls, position: int, width: int):
        if position < 0:
            return 0
        if position >= width:
            return width - 1
        return position

    @classmethod
    def convert_circular_position(cls, position: int, width: int):
        return position % width

    @classmethod
    def convert_fall_position(cls, position: int, width: int):
        return position
