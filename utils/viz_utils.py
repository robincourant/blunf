from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtyping import TensorType

num_points, num_classes = None, None
height, width = None, None


def draw_blueprint(
    point_bundle: TensorType["num_points", 2],
    semantics: TensorType["num_points", 3],
) -> TensorType["height", "width", 3]:
    """Draw the blueprint scatter plot."""
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.scatter(point_bundle[:, 0], point_bundle[:, 1], c=semantics)
    fig.canvas.draw_idle()
    blueprint = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    blueprint = blueprint.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    blueprint = torch.from_numpy(blueprint / 255)
    plt.close("all")
    return blueprint


def draw_confusionmatrix(
    confusion_matrix: TensorType["num_classes", "num_classes"],
) -> TensorType["height", "width", 3]:
    """Draw the blueprint scatter plot."""
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
    h, w = confusion_matrix.shape
    for i, j in product(range(h), range(w)):
        ax.text(
            x=j,
            y=i,
            s=str(round(confusion_matrix[i, j].item() * 100, 1)),
            va="center",
            ha="center",
            size="xx-small",
        )
    ax.set_xlabel("preds")
    ax.set_ylabel("targets")
    fig.canvas.draw_idle()
    conf_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    conf_plot = conf_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    conf_plot = torch.from_numpy(conf_plot / 255)
    plt.close("all")

    return conf_plot
