from pathlib import Path

import click
import numpy as np
import tqdm

from asteroids.agent import AsteroidsAgent
from asteroids.cli.asteroids_cli import main_cli
from asteroids.cli.common_flags import (
    chance_option,
    growth_option,
    height_option,
    width_option,
)
from asteroids.env import AsteroidsEnv
from asteroids.history import HistoryPoint
from asteroids.models import update_target
from asteroids.plotting import plot_all


@main_cli.command("train")
@width_option
@height_option
@chance_option
@growth_option
@click.option("-e", "--episodes", type=int, default=10_000)
@click.option("-b", "--batch-size", type=int, default=64)
@click.option("-w", "--window-size", type=int, default=50)
@click.option("--epsilon", type=float, default=1e-5)
@click.option("-g", "--gamma", type=float, default=0.99)
@click.option("-t", "--tau", type=float, default=0.99)
@click.option("--explore-factor", type=float, default=1)
@click.option("--explore-decay", type=float, default=0.9999)
@click.option("-l", "--learning-rate", type=float, default=0.001)
@click.option("--max-episode-moves", type=int, default=100)
@click.option("-m", "--moves-stop", type=int, default=80)
@click.option("--checkpoint", type=int, default=1_000)
def train_cli(
    width: int,
    height: int,
    chance: float,
    growth: float,
    episodes: int,
    batch_size: int,
    window_size: int,
    epsilon: float,
    gamma: float,
    tau: float,
    explore_factor: float,
    explore_decay: float,
    learning_rate: float,
    max_episode_moves: int,
    moves_stop: int,
    checkpoint: int,
):
    env = AsteroidsEnv(
        width=width,
        height=height,
        start_asteroids_chance=chance,
        asteroids_chance_growth=growth,
    )
    agent = AsteroidsAgent(env=env, batch_size=batch_size, learning_rate=learning_rate)
    history = []
    plots_dir = Path.cwd() / "plots"
    plots_dir.mkdir(exist_ok=True)
    model_path = Path.cwd() / "critic_model.hdf5"
    click.echo(agent.critic.summary())
    with tqdm.trange(episodes) as bar:
        for ep in bar:
            agent.run_episode(
                max_episode_moves=max_episode_moves,
                explore_factor=explore_factor,
                epsilon=epsilon,
            )
            loss = agent.learn(gamma)
            history.append(HistoryPoint.from_env(env=env, loss=loss))
            update_target(target=agent.target_critic, model=agent.critic, tau=tau)
            explore_factor *= explore_decay
            if explore_factor < 1e-5:
                explore_factor = 0

            latest_history = history[-window_size:]
            means_dict = {
                field: np.mean(
                    [getattr(history_point, field) for history_point in latest_history]
                )
                for field in HistoryPoint.fields()
            }
            moves_mean = means_dict["moves"]
            bar.set_description(
                f"Loss mean: {means_dict['loss']:.2f}, "
                f"Moves mean: {moves_mean :.2f}, "
                f"Explore factor: {explore_factor:.2f}"
            )
            if len(history) > window_size and moves_mean >= moves_stop:
                break
            if (ep + 1) % checkpoint == 0:
                plot_all(history=history, window=window_size, output_dir=plots_dir)
                agent.target_critic.save_weights(model_path)
                click.echo(f"Saved model checkpoint in {model_path}")

    plot_all(history=history, window=window_size, output_dir=plots_dir)
    agent.target_critic.save_weights(model_path)
    click.echo(f"Final model was saved in {model_path}")
