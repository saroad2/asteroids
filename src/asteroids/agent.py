import numpy as np
import tensorflow as tf
from keras import Model, layers
from keras.losses import Huber
from keras.optimizers import Adam

from asteroids.action import Action
from asteroids.buffer import Buffer
from asteroids.env import AsteroidsEnv


class AsteroidsAgent:
    def __init__(
        self,
        env: AsteroidsEnv,
        batch_size: int,
        epsilon: float,
        gamma: float,
        explore_factor: float,
        learning_rate: float,
        max_episode_moves: int,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.explore_factor = explore_factor
        self.max_episode_moves = max_episode_moves

        self.buffer = Buffer(
            state_shape=self.env.state_shape,
            batch_size=batch_size,
        )
        self.critic = self.get_critic()
        self.target_critic = self.get_critic()
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

    def run_episode(self):
        state = self.reset()
        states = []
        actions = []
        rewards = []
        next_states = []
        for i in range(self.max_episode_moves):
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action.to_vector())
            rewards.append(reward)
            next_states.append(next_state)
            if done:
                break
            state = next_state

        for state, action, reward, next_state in zip(
            states, actions, rewards, next_states
        ):
            self.buffer.record(
                state=state, action=action, reward=reward, next_state=next_state
            )

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
        state, action, rewards, next_states = self.buffer.batch()
        next_actions = np.array(
            [self.get_action(state).to_vector() for state in next_states]
        )
        with tf.GradientTape() as tape:
            actual_values = self.critic([state, action])
            next_values = self.target_critic([next_states, next_actions])
            expected_values = rewards + self.gamma * next_values
            loss = self.loss_func(actual_values, expected_values)

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        return loss

    def update_target(self, tau):
        for a_target, a in zip(self.target_critic.weights, self.critic.weights):
            a_target.assign(tau * a_target + (1 - tau) * a)

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
