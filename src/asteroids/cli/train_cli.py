from pathlib import Path
from typing import Optional

import click
import numpy as np
import tqdm

from asteroids.agent import AsteroidsAgent
from asteroids.cli.asteroids_cli import main_cli
from asteroids.cli.common_flags import (
    chance_option,
    edge_policy_option,
    growth_option,
    height_option,
    model_suffix_option,
    star_option,
    width_option,
)
from asteroids.constants import MIN_EXPLORE_FACTOR
from asteroids.edge_policy import EdgePolicy
from asteroids.env import AsteroidsEnv
from asteroids.history import HistoryPoint
from asteroids.models import update_target
from asteroids.plotting import plot_all


@main_cli.command("train")
@width_option
@height_option
@chance_option
@growth_option
@star_option
@edge_policy_option
@model_suffix_option
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
@click.option("--score-stop", type=float, default=12)
@click.option("--checkpoint", type=int, default=1_000)
@click.option("--update-model", is_flag=True, default=False)
def train_cli(
    width: int,
    height: int,
    edge_policy: EdgePolicy,
    model_suffix: Optional[str],
    chance: float,
    growth: float,
    star: float,
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
    score_stop: float,
    checkpoint: int,
    update_model: bool,
):
    env = AsteroidsEnv(
        width=width,
        height=height,
        edge_policy=edge_policy,
        start_asteroids_chance=chance,
        asteroids_chance_growth=growth,
        star_chance=star,
    )
    agent = AsteroidsAgent(env=env, batch_size=batch_size, learning_rate=learning_rate)
    history = []
    plots_dir = Path.cwd() / "plots"
    models_directory = Path.cwd() / "models"
    if update_model:
        agent.load_models(models_directory=models_directory, suffix=model_suffix)
    click.echo(agent.critic.summary())
    max_score = 0
    start_explore_factor = explore_factor
    with tqdm.trange(episodes) as bar:
        for ep in bar:
            agent.run_episode(
                max_episode_moves=max_episode_moves,
                explore_factor=explore_factor,
                epsilon=epsilon,
            )
            loss = agent.learn(gamma)
            history.append(
                HistoryPoint.from_env(env=env, loss=loss, explore_factor=explore_factor)
            )
            update_target(target=agent.target_critic, model=agent.critic, tau=tau)

            latest_history = history[-window_size:]
            means_dict = {
                field: np.mean(
                    [getattr(history_point, field) for history_point in latest_history]
                )
                for field in HistoryPoint.fields()
            }
            scores_mean = means_dict["score"]
            bar.set_description(
                f"Loss: {means_dict['loss']:.2f}, "
                f"Moves: {means_dict['moves'] :.2f}, "
                f"Score: {scores_mean :.2f}, "
                f"Explore: {explore_factor:.2e}"
            )
            if scores_mean > 0.6 * max_score:
                explore_factor = max(explore_factor * explore_decay, MIN_EXPLORE_FACTOR)
            elif explore_factor < start_explore_factor:
                explore_factor /= explore_decay
            if len(history) > window_size:
                if scores_mean > max_score:
                    agent.save_models(models_directory, suffix="best")
                    max_score = scores_mean
                    bar.set_postfix_str(f"Best: {max_score:.2f}")
                if scores_mean >= score_stop:
                    break
            if (ep + 1) % checkpoint == 0:
                agent.save_models(models_directory)
                plot_all(history=history, window=window_size, output_dir=plots_dir)

    agent.save_models(models_directory)
    plot_all(history=history, window=window_size, output_dir=plots_dir)
