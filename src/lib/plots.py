"""
Plot library for visualizing the neural network training progress

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from . import paths

color_trained = "dimgrey"

font = FontProperties()
font.set_size(20)

"""
def plot_all_layers(x):
    """
    Function plots all performance of the model with modified layers in one figure

    :param x: data for x-axis (interpolation coefficient)
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Validation loss")
    ax.set_xlabel(r"$\alpha$")

    ax2.spines["right"].set_visible(False)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel(r"$\alpha$")

    files = [file for file in os.listdir(paths.layers) if not re.search("distance", file) and not re.search("q", file)]
    for file in files:
        lab = file.split('_')[-1][0:-4]
        if re.search("loss", file):
            ax.plot(x, np.loadtxt(os.path.join(paths.layers, file)), label=lab, lw=1)
        if re.search("acc", file):
            ax2.plot(x, np.loadtxt(os.path.join(paths.layers, file)), lw=1)

    ax.plot(x, np.loadtxt(paths.loss_interpolated_model_path), label="all", color=color_trained, linewidth=1)
    ax.axvline(x=0, color='black', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.axvline(x=1, color='black', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.set_ylim(bottom=0)

    ax2.plot(x, np.loadtxt(paths.accuracy_interpolated_model_path), color=color_trained, linewidth=1)
    ax2.axvline(x=0, color='black', linestyle=':', linewidth=0.8, alpha=0.8)
    ax2.axvline(x=1, color='black', linestyle=':', linewidth=0.8, alpha=0.8)
    ax2.set_ylim(0, 100)

    fig.legend()
    fig.subplots_adjust(bottom=0.17)

    plt.savefig(os.path.join(paths.interpolation_img, f"all_({paths.name}).pdf"), format="pdf")

    plt.close("all")
"""