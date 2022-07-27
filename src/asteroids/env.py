from typing import List, Optional

import gym
import numpy as np
import pygame
from scipy.stats import poisson

from asteroids.action import Action
from asteroids.constants import BLACK, BLOCK_SIZE, BLUE, RED, WHITE


class AsteroidsEnv(gym.Env):

    live_reward = 0.1
    lost_penalty = 1

    def __init__(
        self,
        width: int,
        height: int,
        start_asteroids_chance: float,
        asteroids_chance_growth: float,
        init_pygame: bool = False,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.start_asteroids_chance = start_asteroids_chance
        self.asteroids_chance_growth = asteroids_chance_growth
        self.player_position = self.width // 2
        self.asteroids: List[np.ndarray] = []
        self.actions_count: np.ndarray = np.zeros(shape=(len(Action)))
        self.score: float = 0
        if init_pygame:
            screen_width, screen_height = (
                self.width * BLOCK_SIZE,
                self.height * BLOCK_SIZE,
            )
            self.screen = pygame.display.set_mode([screen_width, screen_height])
            self.font = pygame.font.SysFont("Ariel", 24)
        else:
            self.screen = None
            self.font = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        self.player_position = self.width // 2
        self.asteroids = []
        self.actions_count = np.zeros(shape=(len(Action)))
        self.score = 0
        return self.state

    @property
    def state_shape(self):
        return self.width, self.height, 2

    @property
    def state(self):
        state = np.zeros(shape=self.state_shape)
        state[self.player_position, 0, 0] = 1
        for asteroid in self.asteroids:
            x, y = asteroid
            state[x, y, 1] = 1
        return state

    @property
    def lost(self):
        for asteroid in self.asteroids:
            if asteroid[1] == 0 and asteroid[0] == self.player_position:
                return True
        return False

    @property
    def moves(self):
        return np.sum(self.actions_count)

    @property
    def entropy(self):
        moves = self.moves
        if moves == 0:
            return 0
        entropy = 0
        for i in range(len(Action)):
            p = self.actions_count[i]
            if p == 0:
                continue
            p /= moves
            entropy -= p * np.log(p)
        return entropy

    def step(self, action: Action):
        self.player_position = action.update_position(
            position=self.player_position, width=self.width
        )
        self.update_enemies()
        lost = self.lost
        reward = -self.lost_penalty if lost else self.live_reward
        self.score += reward
        self.actions_count[action.value] += 1
        return self.state, reward, lost, {}

    def update_enemies(self):
        new_asteroids = []
        for asteroid in self.asteroids:
            asteroid[1] -= 1
            if asteroid[1] < 0:
                continue
            new_asteroids.append(asteroid)
        new_asteroids.extend(self.generate_asteroids())
        self.asteroids = new_asteroids

    def generate_asteroids(self):
        mean_num_of_asteroids = self.start_asteroids_chance * np.exp(
            self.moves * self.asteroids_chance_growth
        )
        num_of_asteroids = min(poisson.rvs(mean_num_of_asteroids), self.width)
        return [
            np.array([pos, self.height - 1])
            for pos in np.random.choice(
                self.width, size=num_of_asteroids, replace=False
            )
        ]

    def render(self, mode="human"):
        if self.screen is None:
            return
        self.screen.fill(WHITE)
        self.draw_item(self.player_position, 0, color=BLUE)
        for asteroid in self.asteroids:
            x, y = asteroid
            self.draw_item(x, y, color=RED)
        img = self.font.render(
            f"Score: {self.score:.2f}, Moves: {self.moves}", False, BLACK
        )
        rect = img.get_rect()
        rect.midtop = (BLOCK_SIZE * self.width // 2, 0)
        self.screen.blit(img, rect)

    def draw_item(self, x, y, color):
        pygame.draw.rect(
            self.screen,
            color=color,
            rect=(
                BLOCK_SIZE * x,
                BLOCK_SIZE * (self.height - y - 1),
                BLOCK_SIZE,
                BLOCK_SIZE,
            ),
        )
