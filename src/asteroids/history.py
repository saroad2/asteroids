from dataclasses import dataclass, fields

from asteroids.env import AsteroidsEnv


@dataclass
class HistoryPoint:
    moves: int
    score: float
    loss: float

    @classmethod
    def fields(cls):
        return [f.name for f in fields(cls)]

    @classmethod
    def from_env(cls, env: AsteroidsEnv, loss: float):
        return HistoryPoint(moves=env.moves, score=env.score, loss=loss)
