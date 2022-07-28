from pathlib import Path

import pygame

from asteroids.agent import AsteroidsAgent
from asteroids.cli.asteroids_cli import main_cli
from asteroids.cli.common_flags import (
    chance_option,
    edge_policy_option,
    growth_option,
    height_option,
    model_suffix_option,
    star_option,
    width_option,
)
from asteroids.env import AsteroidsEnv


@main_cli.command("auto-play")
@width_option
@height_option
@chance_option
@growth_option
@star_option
@edge_policy_option
@model_suffix_option
def auto_play_cli(width, height, chance, growth, star, edge_policy, model_suffix):
    pygame.init()

    env = AsteroidsEnv(
        width=width,
        height=height,
        edge_policy=edge_policy,
        start_asteroids_chance=chance,
        asteroids_chance_growth=growth,
        star_chance=star,
        init_pygame=True,
    )
    agent = AsteroidsAgent(env=env, batch_size=0, learning_rate=0)
    agent.load_models(Path.cwd() / "models", suffix=model_suffix)
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
        state = env.state
        critic_values = agent.get_critic_values(state, use_target=True)
        env.render_chances(*critic_values)
        if not env.lost:
            action = agent.get_action(
                state=state, explore_factor=0, epsilon=0, use_target=True
            )
            env.step(action)

        clock.tick(5)
        pygame.display.flip()
