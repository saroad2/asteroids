from pathlib import Path

import pygame

from asteroids.agent import AsteroidsAgent
from asteroids.cli.asteroids_cli import main_cli
from asteroids.cli.common_flags import (
    chance_option,
    growth_option,
    height_option,
    width_option,
)
from asteroids.env import AsteroidsEnv


@main_cli.command("auto-play")
@width_option
@height_option
@chance_option
@growth_option
def auto_play_cli(width, height, chance, growth):
    pygame.init()

    env = AsteroidsEnv(
        width=width,
        height=height,
        start_asteroids_chance=chance,
        asteroids_chance_growth=growth,
        init_pygame=True,
    )
    agent = AsteroidsAgent(
        env=env,
        batch_size=0,
        learning_rate=0,
        max_episode_moves=0,
    )
    agent.target_critic.load_weights(Path.cwd() / "critic_model.hdf5")
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()

        env.render()
        if not env.lost:
            state = env.state
            action = agent.get_action(
                state=state, explore_factor=0, epsilon=0, use_target=True
            )
            env.step(action)

        clock.tick(5)
        pygame.display.flip()
