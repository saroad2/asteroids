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
from asteroids.plotting import plot_moving_average


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
    moves_stop,
):
    env = AsteroidsEnv(
        width=width,
        height=height,
        start_asteroids_chance=chance,
        asteroids_chance_growth=growth,
    )
    agent = AsteroidsAgent(
        env=env,
        batch_size=batch_size,
        epsilon=epsilon,
        gamma=gamma,
        explore_factor=explore_factor,
        learning_rate=learning_rate,
        max_episode_moves=max_episode_moves,
    )
    losses = []
    moves = []
    with tqdm.trange(episodes) as bar:
        for _ in bar:
            agent.run_episode()
            losses.append(agent.learn())
            moves.append(env.moves)
            agent.update_target(tau)
            agent.explore_factor *= explore_decay
            if agent.explore_factor < 1e-5:
                agent.explore_factor = 0

            loss_mean = np.mean(losses[-window_size:])
            moves_mean = np.mean(moves[-window_size:])
            bar.set_description(
                f"Loss mean: {loss_mean:.2f}, " f"moves_mean: {moves_mean:.2f}"
            )
            if len(moves) > window_size and moves_mean >= moves_stop:
                break

    plots_dir = Path.cwd() / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_moving_average(losses, window=window_size, name="loss", output_dir=plots_dir)
    plot_moving_average(moves, window=window_size, name="moves", output_dir=plots_dir)
    agent.target_critic.save_weights(Path.cwd() / "critic_model.hdf5")
