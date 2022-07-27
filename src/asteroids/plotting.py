from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from asteroids.history import HistoryPoint


def plot_all(history: List[HistoryPoint], window: int, output_dir: Path):
    for field in HistoryPoint.fields():
        values = [getattr(history_point, field) for history_point in history]
        plot_moving_average(
            values=values, window=window, name=field, output_dir=output_dir
        )


def plot_moving_average(values, window, name, output_dir):
    mean_values = np.convolve(values, np.ones(window), mode="valid") / window
    plt.plot(np.arange(window, len(values) + 1), mean_values)
    plt.xlabel("episode")
    plt.ylabel(name)
    plt.title(f"{name} rolling average ({window=})")
    plt.savefig(output_dir / f"{name}.jpg")
    plt.clf()
