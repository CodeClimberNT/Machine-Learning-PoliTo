from Visualizer import Visualizer as Vis
from helper import DatasetImporterHelper as Ds
import numpy as np


def plot_features(x, y, labels_name: dict[int, str]):
    custom_titles = {
        (0, 0): "Histogram of Feature 1",
        (1, 1): "Histogram of Feature 2",
        (0, 1): "Scatter Plot of Feature 1 vs Feature 2",
        (1, 0): "Scatter Plot of Feature 2 vs Feature 1",

        (2, 2): "Histogram of Feature 3",
        (3, 3): "Histogram of Feature 4",
        (2, 3): "Scatter Plot of Feature 3 vs Feature 4",
        (3, 2): "Scatter Plot of Feature 4 vs Feature 3",

        (4, 4): "Histogram of Feature 5",
        (5, 5): "Histogram of Feature 6",
        (4, 5): "Scatter Plot of Feature 5 vs Feature 6",
        (5, 4): "Scatter Plot of Feature 6 vs Feature 5"
    }

    custom_x_labels = {
        (0, 0): "Value of Feature 1",
        (1, 1): "Value of Feature 2",
        (0, 1): "Value of Feature 1",
        (1, 0): "Value of Feature 2",

        (2, 2): "Value of Feature 3",
        (3, 3): "Value of Feature 4",
        (2, 3): "Value of Feature 3",
        (3, 2): "Value of Feature 4",

        (4, 4): "Value of Feature 5",
        (5, 5): "Value of Feature 6",
        (4, 5): "Value of Feature 5",
        (5, 4): "Value of Feature 6"
    }

    custom_y_labels = {
        (0, 0): "Frequency of Feature 1",
        (1, 1): "Frequency of Feature 2",
        (0, 1): "Value of Feature 2",
        (1, 0): "Value of Feature 1",

        (2, 2): "Frequency of Feature 3",
        (3, 3): "Frequency of Feature 4",
        (2, 3): "Value of Feature 4",
        (3, 2): "Value of Feature 3",

        (4, 4): "Frequency of Feature 5",
        (5, 5): "Frequency of Feature 6",
        (4, 5): "Value of Feature 6",
        (5, 4): "Value of Feature 5"
    }

    Vis.project1_scatter_hist_pair_features(x, y, labels_name=labels_name, custom_titles=custom_titles,
                                            custom_x_labels=custom_x_labels, custom_y_labels=custom_y_labels,
                                            show_legend=True)


def calculate_mean_and_variance(x: np.ndarray, save_as_txt: bool = False) -> None:
    for i in range(x.shape[0]):
        print(f"Feature {i + 1}:")
        print(f"Mean: {x[i].mean()}")
        print(f"Variance: {x[i].var()}")
        print()

        if save_as_txt:
            with open("../project_solution/project1/mean_variance.txt", "a") as f:
                f.write(f"Feature {i + 1}:\n")
                f.write(f"Mean: {x[i].mean()}\n")
                f.write(f"Variance: {x[i].var()}\n\n")


def main(plotting: bool = False, meaning: bool = False) -> None:
    x, y = Ds.load_train_project()

    labels_name: dict[int, str] = {0: 'True', 1: 'False'}

    if plotting:
        plot_features(x, y, labels_name)

    if meaning:
        calculate_mean_and_variance(x, save_as_txt=True)


if __name__ == "__main__":
    main(plotting=True, meaning=False)
