import click

from asteroids.constants import (
    DEFAULT_ASTEROIDS_CHANCE_GROWTH,
    DEFAULT_ASTEROIDS_START_CHANCE,
    DEFAULT_HEIGHT,
    DEFAULT_STAR_CHANCE,
    DEFAULT_WIDTH,
)
from asteroids.edge_policy import EdgePolicy

width_option = click.option("-w", "--width", type=int, default=DEFAULT_WIDTH)
height_option = click.option("-h", "--height", type=int, default=DEFAULT_HEIGHT)
chance_option = click.option(
    "-c", "--chance", type=float, default=DEFAULT_ASTEROIDS_START_CHANCE
)
growth_option = click.option(
    "-g", "--growth", type=float, default=DEFAULT_ASTEROIDS_CHANCE_GROWTH
)
star_option = click.option("-s", "--star", type=float, default=DEFAULT_STAR_CHANCE)
edge_policy_option = click.option(
    "--edge-policy",
    type=click.Choice(EdgePolicy.names(), case_sensitive=False),
    default=EdgePolicy.WALL.name,
    callback=lambda ctx, param, value: EdgePolicy[value],
)
model_suffix_option = click.option("--model-suffix", type=str)
