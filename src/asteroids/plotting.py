from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from asteroids.history import HistoryPoint


def plot_all(history: List[HistoryPoint], output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    for field in HistoryPoint.fields():
        values = [getattr(history_point, field) for history_point in history]
        plot_smoothly(values=values, name=field, output_dir=output_dir)


def plot_smoothly(values, name, output_dir):
    smooth_values = savgol_filter(values, window_length=len(values) // 10, polyorder=3)
    plt.plot(np.arange(len(smooth_values)), smooth_values)
    plt.xlabel("episode")
    plt.ylabel(name)
    plt.title(f"{name} smoothed")
    plt.savefig(output_dir / f"{name}.jpg")
    plt.clf()
