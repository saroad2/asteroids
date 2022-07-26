import click
import pygame

from asteroids.action import Action
from asteroids.cli.asteroids_cli import main_cli
from asteroids.constants import (
    DEFAULT_ASTEROIDS_CHANCE_GROWTH,
    DEFAULT_ASTEROIDS_START_CHANCE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
)
from asteroids.env import AsteroidsEnv


@main_cli.command("play")
@click.option("-w", "--width", type=int, default=DEFAULT_WIDTH)
@click.option("-h", "--height", type=int, default=DEFAULT_HEIGHT)
@click.option("-c", "--chance", type=float, default=DEFAULT_ASTEROIDS_START_CHANCE)
@click.option("-g", "--growth", type=float, default=DEFAULT_ASTEROIDS_CHANCE_GROWTH)
def play_cli(width, height, chance, growth):
    pygame.init()

    env = AsteroidsEnv(
        width=width,
        height=height,
        start_asteroids_chance=chance,
        asteroids_chance_growth=growth,
        init_pygame=True,
    )
    running = True
    clock = pygame.time.Clock()
    while running:
        action = Action.NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                if event.key == pygame.K_LEFT:
                    action = Action.LEFT
                if event.key == pygame.K_RIGHT:
                    action = Action.RIGHT

        env.render()
        if not env.lost:
            env.step(action)

        clock.tick(5)
        pygame.display.flip()
