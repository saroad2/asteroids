from collections import deque
from typing import Optional

import gym
import numpy as np
import pygame
from scipy.stats import poisson

from asteroids.action import Action
from asteroids.constants import BLACK, BLOCK_SIZE, BLUE, GREEN, RED, WHITE, YELLOW
from asteroids.edge_policy import EdgePolicy


class AsteroidsEnv(gym.Env):

    star_reward = 2
    live_reward = 0.1
    lost_penalty = 1

    def __init__(
        self,
        width: int,
        height: int,
        edge_policy: EdgePolicy,
        start_asteroids_chance: float,
        asteroids_chance_growth: float,
        star_chance: float,
        init_pygame: bool = False,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.edge_policy = edge_policy
        self.start_asteroids_chance = start_asteroids_chance
        self.asteroids_chance_growth = asteroids_chance_growth
        self.star_chance = star_chance

        self.player_position = self.width // 2
        self.asteroids: deque[np.ndarray] = deque()
        self.stars: deque[np.ndarray] = deque()
        self.actions_count: np.ndarray = np.zeros(shape=(len(Action)))
        self.star_hits = 0
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
        self.asteroids.clear()
        self.stars.clear()
        self.actions_count = np.zeros(shape=(len(Action)))
        self.star_hits = 0
        self.score = 0
        return self.state

    @property
    def state_shape(self):
        return self.width, self.height, 3

    @property
    def player_in_board(self):
        return 0 <= self.player_position < self.width

    @property
    def state(self):
        state = np.zeros(shape=self.state_shape)
        if self.lost:
            return state
        state[self.player_position, 0, 0] = 1
        for asteroid in self.asteroids:
            x, y = asteroid
            state[x, y, 1] = 1
        for star in self.stars:
            x, y = star
            state[x, y, 2] = 1
        return state

    @property
    def lost(self):
        if not self.player_in_board:
            return True
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
            position=self.player_position,
            width=self.width,
            edge_policy=self.edge_policy,
        )
        self.update_environment()
        hit_star = False
        if len(self.stars) > 0 and self.stars[-1][1] == 0:
            last_star = self.stars.pop()
            if last_star[0] == self.player_position:
                hit_star = True
                self.star_hits += 1
        reward = self.get_reward(hit_star=hit_star)
        self.score += reward
        self.actions_count[action.value] += 1
        return self.state, reward, self.lost, {}

    def update_environment(self):
        self.update_falling_deque(self.asteroids)
        self.update_falling_deque(self.stars)
        new_asteroids_positions = self.generate_asteroids_positions()
        self.asteroids.extendleft(
            [np.array([pos, self.height - 1]) for pos in new_asteroids_positions]
        )
        if np.random.uniform() > self.star_chance:
            return
        star_pos_options = [
            i for i in range(self.width) if i not in new_asteroids_positions
        ]
        if len(star_pos_options) == 0:
            return
        star_pos = np.random.choice(star_pos_options)
        self.stars.appendleft(np.array([star_pos, self.height - 1]))

    def generate_asteroids_positions(self):
        mean_num_of_asteroids = self.start_asteroids_chance * np.exp(
            self.moves * self.asteroids_chance_growth
        )
        num_of_asteroids = min(poisson.rvs(mean_num_of_asteroids), self.width)
        return np.random.choice(self.width, size=num_of_asteroids, replace=False)

    def get_reward(self, hit_star):
        if hit_star:
            return self.star_reward
        if self.lost:
            return -self.lost_penalty
        return self.live_reward

    def render(self, mode="human"):
        if self.screen is None:
            return
        self.screen.fill(WHITE)
        for star in self.stars:
            x, y = star
            self.draw_item(x, y, color=YELLOW)
        self.draw_item(self.player_position, 0, color=BLUE)
        for asteroid in self.asteroids:
            x, y = asteroid
            self.draw_item(x, y, color=RED)
        img = self.font.render(
            f"Score: {self.score:.2f}, "
            f"Moves: {self.moves}, "
            f"Star hits: {self.star_hits}",
            False,
            BLACK,
        )
        rect = img.get_rect()
        rect.midtop = (BLOCK_SIZE * self.width // 2, 0)
        self.screen.blit(img, rect)

    def render_chances(self, left, middle, right):
        max_index = np.argmax([left, middle, right])
        left_color = GREEN if max_index == 0 else BLACK
        img_left = self.font.render(f"{left:.3f}", False, left_color)
        rect_left = img_left.get_rect()
        rect_left.midleft = (0, BLOCK_SIZE * self.height // 2)
        self.screen.blit(img_left, rect_left)

        center_color = GREEN if max_index == 1 else BLACK
        img_center = self.font.render(f"{middle:.3f}", False, center_color)
        rect_center = img_center.get_rect()
        rect_center.center = (
            BLOCK_SIZE * self.width // 2,
            BLOCK_SIZE * self.height // 2,
        )
        self.screen.blit(img_center, rect_center)

        right_color = GREEN if max_index == 2 else BLACK
        img_right = self.font.render(f"{right:.3f}", False, right_color)
        rect_right = img_right.get_rect()
        rect_right.midright = (BLOCK_SIZE * self.width, BLOCK_SIZE * self.height // 2)
        self.screen.blit(img_right, rect_right)

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

    @classmethod
    def update_falling_deque(cls, falling_deque: deque):
        for obj in falling_deque:
            obj[1] -= 1
        while len(falling_deque) > 0 and falling_deque[-1][1] < 0:
            falling_deque.pop()
