from matplotlib import pyplot as plt
import numpy as np


def load(file_path: str, features_cols: tuple) -> tuple[np.ndarray, np.ndarray]:
    # Load the array from the text file
    x = np.loadtxt(file_path, delimiter=",", usecols=features_cols, dtype=float)
    y = np.loadtxt(file_path, delimiter=",", usecols=features_cols[-1] + 1, dtype=str)
    # Return the array
    return x, y


def plot(x: np.ndarray, y: np.ndarray, features: list[str]) -> None:
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            plt.figure()
            for label in np.unique(y):
                if '-' in label:
                    classes = label.split('-')[1].capitalize()  # capitalize the first character
                else:
                    classes = label.capitalize()
                if i != j:
                    plt.scatter(x[y == label, i], x[y == label, j], label=classes, alpha=0.5)
                else:
                    plt.hist(x[y == label, i], density=True, label=classes, alpha=0.5)
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.legend()
            plt.show()


def m_col(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], 1)


def m_row(x: np.ndarray) -> np.ndarray:
    return x.reshape(1, x.shape[0])


def main() -> None:
    # Specify the file path
    file_path: str = "./iris.csv"

    # Load features and label
    [x, y] = load(file_path, (0, 1, 2, 3))

    print(x.shape)
    print(y.shape)

    features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    plot(x, y, features)

    mu = x.mean(axis=1).reshape(x.shape[0], 1)

    xc = x - mu

    plot(x, y, features)

    cov = (xc @ xc.T) / float(x.shape[1])

    print(x.shape)

    var = x.var(1)
    std = x.std(1)

    print(var.shape)


if __name__ == '__main__':
    main()
