from dataclasses import dataclass, fields

from asteroids.env import AsteroidsEnv


@dataclass
class HistoryPoint:
    moves: int
    star_hits: int
    score: float
    entropy: float
    loss: float
    explore_factor: float

    @classmethod
    def fields(cls):
        return [f.name for f in fields(cls)]

    @classmethod
    def from_env(cls, env: AsteroidsEnv, loss: float, explore_factor: float):
        return HistoryPoint(
            moves=env.moves,
            star_hits=env.star_hits,
            score=env.score,
            entropy=env.entropy,
            loss=loss,
            explore_factor=explore_factor,
        )
