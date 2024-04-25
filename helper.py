import numpy as np
from matplotlib import pyplot as plt


def load_txt(
	file_path: str,
	features_type: type = float,
	return_labels_in_data_matrix: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

	# Load the array from the text file and convert all but last column to float

	data = np.loadtxt(file_path, delimiter=",", dtype=str)

	if return_labels_in_data_matrix:
		return data

	# Extract features (all but last column)

	x = data[:, :-1].astype(features_type)

	# Extract labels (last column)

	y = data[:, -1]

	return x, y


def plot_matrix(
	matrix: np.ndarray,
	labels: np.ndarray,
	*,
	title: str,
	xlabel: str,
	ylabel: str,
	grid: bool = True,
	invert_yaxis: bool = False,
) -> None:

	plt.figure()

	unique_labels = np.unique(labels)

	colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

	for label, color in zip(unique_labels, colors):

		indices = np.where(labels == label)
		plt.scatter(matrix[indices, 0], matrix[indices, 1], color=color, label=label)

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)


	plt.legend()

	plt.grid(grid)

	if invert_yaxis:
		plt.gca().invert_yaxis()

	plt.show()


def v_col(x):

	return x.reshape((x.size, 1))


def v_row(x):

	return x.reshape((1, x.size))


def center_matrix(matrix: np.ndarray) -> np.ndarray:

	# compute mean over the rows

	mu = matrix.mean(0)

	print(f"Mean: \n{mu}")

	return matrix - mu


def check_solution() -> None:

	raise NotImplementedError


def cv_matrix(matrix: np.ndarray) -> np.ndarray:

	centered_matrix: np.ndarray = center_matrix(matrix)

	return np.dot(centered_matrix.T, centered_matrix) / float(matrix.shape[0])
