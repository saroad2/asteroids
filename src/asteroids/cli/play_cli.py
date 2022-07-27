import pygame

from asteroids.action import Action
from asteroids.cli.asteroids_cli import main_cli
from asteroids.cli.common_flags import (
    chance_option,
    edge_policy_option,
    growth_option,
    height_option,
    width_option,
)
from asteroids.env import AsteroidsEnv


@main_cli.command("play")
@width_option
@height_option
@chance_option
@growth_option
@edge_policy_option
def play_cli(width, height, chance, growth, edge_policy):
    pygame.init()

    env = AsteroidsEnv(
        width=width,
        height=height,
        edge_policy=edge_policy,
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
