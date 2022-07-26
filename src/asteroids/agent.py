import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from asteroids.action import Action
from asteroids.buffer import Buffer
from asteroids.env import AsteroidsEnv


class AsteroidsAgent:
    def __init__(
        self,
        env: AsteroidsEnv,
        batch_size: int,
        epsilon: float,
        tau: float,
        gamma: float,
        explore_factor: float,
        learning_rate: float,
        max_episode_moves: int,
    ):
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.explore_factor = explore_factor
        self.max_episode_moves = max_episode_moves

        self.buffer = Buffer(
            state_shape=self.env.state_shape,
            batch_size=batch_size,
        )
        self.critic = self.get_critic()
        self.optimizer = Adam(learning_rate=learning_rate)
        self.loss_func = Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.counts = np.ones(shape=len(Action))

    @property
    def counts_sum(self):
        return np.sum(self.counts)

    def reset(self):
        self.counts = np.ones(shape=len(Action))
        return self.env.reset()

    def run_episode(self):
        state = self.reset()
        states = []
        actions = []
        rewards = []
        for i in range(self.max_episode_moves):
            action = self.get_action(state)
            state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action.to_vector())
            rewards.append(reward)
            if done:
                break
        rewards = rewards[::-1]
        values = [reward * np.power(self.gamma, i) for i, reward in enumerate(rewards)]
        values = np.cumsum(values)
        for state, action, value in zip(states, actions, values):
            self.buffer.record(state, action, value)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            action_index = np.random.choice(len(Action))
            return Action(action_index)
        state_tf = state.reshape((-1, *self.env.state_shape))
        state_tf = np.repeat(state_tf, repeats=len(Action), axis=0)
        action_tf = np.identity(len(Action))
        critic_value = self.critic([state_tf, action_tf])
        critic_value = np.squeeze(critic_value)
        counts_sum_log = np.log(self.counts_sum)
        explore_value = self.explore_factor * counts_sum_log / self.counts
        action_index = np.argmax(critic_value + explore_value)
        return Action(action_index)

    def learn(self):
        state, action, values = self.buffer.batch()
        with tf.GradientTape() as tape:
            actual_values = self.critic([state, action])
            loss = self.loss_func(actual_values, values)

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        return loss

    def get_critic(self) -> Model:
        state_input = layers.Input(shape=self.env.state_shape)
        state_out = layers.Conv2D(16, (3, 3), activation="relu")(state_input)
        state_out = layers.MaxPool2D()(state_out)
        state_out = layers.Conv2D(32, (2, 2), activation="relu")(state_out)
        state_out = layers.MaxPool2D()(state_out)
        state_out = layers.Flatten()(state_out)
        state_out = layers.Dense(64, activation="relu")(state_out)

        action_input = layers.Input(shape=(len(Action),))
        action_out = layers.Dense(32, activation="relu")(action_input)
        action_out = layers.Dense(64, activation="relu")(action_out)

        concat = layers.Concatenate()([state_out, action_out])
        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(1, activation="relu")(out)

        return Model(inputs=[state_input, action_input], outputs=out)
