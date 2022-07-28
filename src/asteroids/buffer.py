import numpy as np

from asteroids.action import Action
from asteroids.constants import BUFFER_SIZE


class Buffer:
    def __init__(self, state_shape, batch_size: int, buffer_size: int = BUFFER_SIZE):
        self.state_buffer = np.zeros(shape=(buffer_size, *state_shape))
        self.action_buffer = np.zeros(shape=(buffer_size, len(Action)))
        self.reward_buffer = np.zeros(shape=buffer_size)
        self.next_state_buffer = np.zeros(shape=(buffer_size, *state_shape))
        self.done_buffer = np.zeros(shape=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.count = 0

    def record(self, state, action, reward, next_state, done):
        i = self.count % self.buffer_size
        self.state_buffer[i] = state
        self.action_buffer[i] = action
        self.reward_buffer[i] = reward
        self.next_state_buffer[i] = next_state
        self.done_buffer[i] = 0 if done else 1
        self.count += 1

    def batch(self):
        max_index = min(self.count, self.buffer_size)
        batch_size = min(self.count, self.batch_size)
        batch_indices = np.random.choice(max_index, size=batch_size, replace=False)
        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]
        done_batch = self.done_buffer[batch_indices]
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
