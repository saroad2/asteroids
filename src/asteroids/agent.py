from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from keras.losses import Huber
from keras.optimizers import Adam

from asteroids.action import Action
from asteroids.buffer import Buffer
from asteroids.env import AsteroidsEnv
from asteroids.models import get_critic


class AsteroidsAgent:
    def __init__(
        self,
        env: AsteroidsEnv,
        batch_size: int,
        learning_rate: float,
    ):
        self.env = env

        self.buffer = Buffer(
            state_shape=self.env.state_shape,
            batch_size=batch_size,
        )
        self.critic = get_critic(env=self.env)
        self.target_critic = get_critic(env=self.env)
        self.target_critic.set_weights(self.critic.get_weights())
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_func = Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.counts = np.ones(shape=len(Action))

    @property
    def counts_sum(self):
        return np.sum(self.counts)

    def reset(self):
        self.counts = np.ones(shape=len(Action))
        return self.env.reset()

    def run_episode(
        self, max_episode_moves: int, explore_factor: float, epsilon: float
    ):
        state = self.reset()
        for i in range(max_episode_moves):
            action = self.get_action(
                state, explore_factor=explore_factor, epsilon=epsilon
            )
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.record(
                state=state,
                action=action.to_vector(),
                reward=reward,
                next_state=next_state,
                done=done,
            )
            if done:
                break
            state = next_state

    def get_action(self, state, explore_factor, epsilon, use_target: bool = False):
        if np.random.uniform() < epsilon:
            action_index = np.random.choice(len(Action))
            return Action(action_index)
        critic_value = self.get_critic_values(state, use_target)
        counts_sum_log = np.log(self.counts_sum)
        explore_value = explore_factor * counts_sum_log / self.counts
        action_index = np.argmax(critic_value + explore_value)
        self.counts[action_index] += 1
        return Action(action_index)

    def get_critic_values(self, state, use_target):
        state_tf = state.reshape((-1, *self.env.state_shape))
        state_tf = np.repeat(state_tf, repeats=len(Action), axis=0)
        action_tf = np.identity(len(Action))
        critic_model = self.target_critic if use_target else self.critic
        critic_value = critic_model([state_tf, action_tf])
        return np.squeeze(critic_value)

    def learn(self, gamma: float):
        state, action, rewards, next_states, done = self.buffer.batch()
        next_actions = np.array(
            [
                self.get_action(
                    state, explore_factor=0, epsilon=0, use_target=True
                ).to_vector()
                for state in next_states
            ]
        )
        next_values = tf.squeeze(self.target_critic([next_states, next_actions]))
        expected_values = rewards + gamma * done * next_values
        with tf.GradientTape() as tape:
            actual_values = tf.squeeze(self.critic([state, action]))
            loss = self.loss_func(actual_values, expected_values)

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        return loss

    def save_models(self, models_directory: Path, suffix: Optional[str] = None):
        models_directory.mkdir(exist_ok=True)
        self.critic.save_weights(
            models_directory / self.model_file_name("critic", suffix=suffix)
        )
        self.target_critic.save_weights(
            models_directory / self.model_file_name("target_critic", suffix=suffix)
        )

    def load_models(self, models_directory: Path, suffix: Optional[str] = None):
        self.critic.load_weights(
            models_directory / self.model_file_name("critic", suffix=suffix)
        )
        self.target_critic.load_weights(
            models_directory / self.model_file_name("target_critic", suffix=suffix)
        )

    def model_file_name(self, prefix: str, suffix: Optional[str] = None):
        name = (
            f"{prefix}_"
            f"{self.env.edge_policy.name.lower()}_"
            f"{self.env.width}_{self.env.height}"
        )
        if suffix is not None:
            name += "_" + suffix
        return f"{name}.hdf5"
