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
        max_episode_moves: int,
    ):
        self.env = env
        self.max_episode_moves = max_episode_moves

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

    def run_episode(self, explore_factor: float, epsilon: float):
        state = self.reset()
        states = []
        actions = []
        rewards = []
        next_states = []
        for i in range(self.max_episode_moves):
            action = self.get_action(
                state, explore_factor=explore_factor, epsilon=epsilon
            )
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

    def get_action(self, state, explore_factor, epsilon, use_target: bool = False):
        if np.random.uniform() < epsilon:
            action_index = np.random.choice(len(Action))
            return Action(action_index)
        state_tf = state.reshape((-1, *self.env.state_shape))
        state_tf = np.repeat(state_tf, repeats=len(Action), axis=0)
        action_tf = np.identity(len(Action))
        critic_model = self.target_critic if use_target else self.critic
        critic_value = critic_model([state_tf, action_tf])
        critic_value = np.squeeze(critic_value)
        counts_sum_log = np.log(self.counts_sum)
        explore_value = explore_factor * counts_sum_log / self.counts
        action_index = np.argmax(critic_value + explore_value)
        return Action(action_index)

    def learn(self, gamma: float):
        state, action, rewards, next_states = self.buffer.batch()
        next_actions = np.array(
            [
                self.get_action(
                    state, explore_factor=0, epsilon=0, use_target=True
                ).to_vector()
                for state in next_states
            ]
        )
        with tf.GradientTape() as tape:
            actual_values = self.critic([state, action])
            next_values = self.target_critic([next_states, next_actions])
            expected_values = rewards + gamma * next_values
            loss = self.loss_func(actual_values, expected_values)

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        return loss
