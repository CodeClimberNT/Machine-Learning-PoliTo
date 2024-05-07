import numpy as np
from matplotlib import pyplot as plt

import logging




class Visualizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_multiple_hist(
        matrix: np.ndarray,
        labels: np.ndarray,
        labels_name: dict[int, str],
        title: str,
        x_label: str,
        y_label: str,
    ) -> None:

        plt.figure()

        unique_labels = np.unique(labels)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        if labels_name and len(labels_name) == len(unique_labels):

            unique_labels = labels_name

        for label, color in zip(unique_labels, colors):

            plt.hist(matrix[:, labels == labels], bins=20, label=labels_name.get(label))

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.legend()

        plt.show()

    @staticmethod
    def plot_scatter_matrix(
        matrix: np.ndarray,
        labels: np.ndarray,
        labels_name: dict[int, str],
        *,
        title: str,
        x_label: str,
        y_label: str,
        grid: bool = True,
        invert_y_axis: bool = False,
        invert_x_axis: bool = False,
    ) -> None:

        plt.figure()

        unique_labels = np.unique(labels)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        if labels_name and len(labels_name) == len(unique_labels):

            unique_labels = labels_name

        for label, color in zip(unique_labels, colors):

            indices = np.where(labels == label)

            plt.scatter(matrix[0, indices], matrix[1, indices], color=color, label=label)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.legend()

        plt.grid(grid)

        if invert_y_axis:
            plt.gca().invert_yaxis()
        if invert_x_axis:
            plt.gca().invert_xaxis()
            
        plt.show()

