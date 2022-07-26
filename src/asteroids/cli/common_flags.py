import click

from asteroids.constants import (
    DEFAULT_ASTEROIDS_CHANCE_GROWTH,
    DEFAULT_ASTEROIDS_START_CHANCE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
)

width_option = click.option("-w", "--width", type=int, default=DEFAULT_WIDTH)

height_option = click.option("-h", "--height", type=int, default=DEFAULT_HEIGHT)

chance_option = click.option(
    "-c", "--chance", type=float, default=DEFAULT_ASTEROIDS_START_CHANCE
)

growth_option = click.option(
    "-g", "--growth", type=float, default=DEFAULT_ASTEROIDS_CHANCE_GROWTH
)
