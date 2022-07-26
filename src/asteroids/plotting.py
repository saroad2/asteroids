import numpy as np
from matplotlib import pyplot as plt


def plot_moving_average(values, window, name, output_dir):
    mean_values = np.convolve(values, np.ones(window), mode="valid") / window
    plt.plot(np.arange(window, len(values) + 1), mean_values)
    plt.xlabel("episode")
    plt.ylabel(name)
    plt.title(f"{name} rolling average ({window=})")
    plt.savefig(output_dir / f"{name}.jpg")
    plt.clf()
