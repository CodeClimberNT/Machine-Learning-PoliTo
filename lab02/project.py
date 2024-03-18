from typing import Any

from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray, dtype


def load_file_into_np(file_path: str, features_cols: tuple) -> list[ndarray[Any, dtype[Any]]]:
    # Load the features array from the text file
    x: ndarray[Any, dtype[float]] = np.loadtxt(file_path, delimiter=",", usecols=features_cols, dtype=float)

    # Load the labels array from the text file
    y: ndarray[Any, dtype[int]] = np.loadtxt(file_path, delimiter=",", usecols=features_cols[-1] + 1, dtype=int)

    # Return the list of features and labels
    return [x, y]


def plot(x, y, label_name: tuple[str], title: str) -> None:
    plt.figure()

    rows: int = x.shape[1]
    cols: int = x.shape[1]

    fig, axs = plt.subplots(rows, cols, figsize=(cols*7, rows*7))
    # fig.suptitle(title)

    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            for label in np.unique(y):
                if i != j:
                    axs[i, j].scatter(x[y == label, i], x[y == label, j], label=label_name[label], alpha=0.5)
                else:
                    axs[i, j].hist(x[y == label, i], density=True, label=label_name[label], alpha=0.5)
                axs[i, j].legend(title='Labels')

    fig.show()


def main():
    # Hardcoded file path
    file_path: str = "./trainData.txt"

    [x, y] = load_file_into_np(file_path, (0, 1, 2, 3, 4, 5))

    features: tuple = ('True', 'False')
    # make a 1x2 plot of the features against each other

    # plot first two features
    plot(x[:, :2], y, features, 'First and Second features')

    # plot next two features
    plot(x[:, 2:4], y, features, 'Third and Fourth features')

    # plot last two features
    plot(x[:, 4:6], y, features, 'Fifth and Sixth features')


if __name__ == "__main__":
    main()
