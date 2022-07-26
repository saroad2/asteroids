from pathlib import Path

import numpy as np
import pygame

from asteroids.action import Action
from asteroids.cli.asteroids_cli import main_cli
from asteroids.cli.common_flags import (
    chance_option,
    growth_option,
    height_option,
    width_option,
)
from asteroids.env import AsteroidsEnv
from asteroids.models import load_critic


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
    critic_model = load_critic(env=env, path=Path.cwd() / "critic_model.hdf5")
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
            state_tf = state.reshape((-1, *env.state_shape))
            state_tf = np.repeat(state_tf, repeats=len(Action), axis=0)
            action_tf = np.identity(len(Action))
            critic_value = critic_model([state_tf, action_tf])
            critic_value = np.squeeze(critic_value)
            action = Action(np.argmax(critic_value))
            env.step(action)

        clock.tick(5)
        pygame.display.flip()
